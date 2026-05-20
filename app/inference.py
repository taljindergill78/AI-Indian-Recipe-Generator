"""
Inference engine for the Indian Kitchen Gradio app.

Handles model loading, recipe generation (streaming), recipe adaptation,
and MLflow trace logging for LLM observability.

INFERENCE_MODE (set via env var):
  "local"  — loads fine-tuned model via transformers + PEFT (requires GPU/CUDA)
  "demo"   — returns pre-written example recipes (for UI testing without GPU)

MLflow Traces:
  Every completed generation is logged to the "qlora-indian-recipe-finetune"
  experiment as a nested run. This populates the Traces tab in mlflow ui.
  We use mlflow.start_run() rather than @mlflow.trace because the generator
  yields tokens incrementally — @mlflow.trace expects a direct return value.

Usage:
    from app.inference import load_model, generate_stream, adapt_stream
    load_model()  # called once at Gradio startup
    for chunk in generate_stream(dish_name="Palak Paneer", diet="Vegetarian", region="North Indian"):
        print(chunk, end="", flush=True)
"""

import os
import time
import random
import mlflow

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")
BASE_MODEL_ID  = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_ID     = "taljindergill78/indian-recipe-llama3.2-qlora"
MLFLOW_EXP     = "qlora-indian-recipe-finetune"

# ZeroGPU: HuggingFace Spaces shared GPU. When the `spaces` package is present,
# we apply @spaces.GPU to the generation function so it gets GPU access per call.
# On local Mac/CPU (no `spaces` package), we fall through to demo mode.
try:
    import spaces as _spaces_module
    _HAS_ZEROGPU = True
except ImportError:
    _spaces_module = None
    _HAS_ZEROGPU = False

SYSTEM_PROMPT = (
    "You are an expert Indian chef with decades of culinary expertise across all regional "
    "Indian cuisines — from the tandoors of North India to the coconut gravies of Kerala, "
    "the mustard-spiced dishes of Bengal to the tamarind-rich curries of Tamil Nadu. "
    "Generate authentic, detailed Indian recipes with clear ingredients and step-by-step "
    "cooking instructions. Always start with the recipe name in bold (**Name**), followed "
    "by **Ingredients:** and **Instructions:** sections."
)

mlflow.set_experiment(MLFLOW_EXP)

# Module-level model references (loaded once at startup)
_tokenizer = None
_model = None


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model() -> bool:
    """
    Load the fine-tuned model and tokenizer into module globals.

    Behaviour by environment:
      ZeroGPU (HF Spaces): model loads lazily on first @spaces.GPU call — this
        function just prints a message and returns True.
      Local + CUDA:  loads model immediately at startup.
      Local, no CUDA: falls back to demo mode automatically.
    """
    global _tokenizer, _model

    if INFERENCE_MODE == "demo":
        print("[inference] Demo mode — skipping model load.")
        return True

    if _HAS_ZEROGPU:
        # On HuggingFace Spaces with ZeroGPU: GPU is not available at startup.
        # The model will be loaded lazily inside the @spaces.GPU decorated function
        # on the first inference call. HF caches weights on disk after first download.
        print("[inference] ZeroGPU detected — model will load on first inference call.")
        return True

    # Local mode: requires a CUDA GPU
    return _load_weights_now()


def _load_weights_now() -> bool:
    """Load weights into GPU memory (called at startup in local mode, lazily in ZeroGPU)."""
    global _tokenizer, _model

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        if not torch.cuda.is_available():
            print("[inference] WARNING: No CUDA GPU detected. Falling back to demo mode.")
            globals()["INFERENCE_MODE"] = "demo"
            return False

        print(f"[inference] Loading tokenizer from {ADAPTER_ID}...")
        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        print(f"[inference] Loading base model {BASE_MODEL_ID} in 4-bit NF4...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
        )

        print(f"[inference] Attaching LoRA adapter from {ADAPTER_ID}...")
        _model = PeftModel.from_pretrained(base, ADAPTER_ID)
        _model.eval()
        print("[inference] Model ready.")
        return True

    except Exception as exc:
        print(f"[inference] Model load failed: {exc}")
        globals()["INFERENCE_MODE"] = "demo"
        return False


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def _build_user_message(
    dish_name: str,
    ingredients: list[str],
    diet: str,
    region: str,
    time_pref: str = "Any Time",
) -> str:
    """Match the training data format: 'Generate a {diet} {region} recipe for {name}'."""
    if dish_name and dish_name.strip():
        # Dish-name mode (matches training format exactly)
        descriptor_parts = []
        if diet and diet != "Any Diet":
            descriptor_parts.append(diet)
        if region and region != "Any Region":
            descriptor_parts.append(region)
        descriptor = " ".join(descriptor_parts) or "Indian"
        article = "an" if descriptor[0].lower() in "aeiou" else "a"
        return f"Generate {article} {descriptor} recipe for {dish_name.strip()}"
    else:
        # Ingredient-based mode
        ing_str = ", ".join(ingredients) if ingredients else "paneer, spinach, tomato"
        parts = []
        if diet and diet != "Any Diet":
            parts.append(diet)
        if region and region != "Any Region":
            parts.append(region)
        parts.append("recipe")

        time_clause = ""
        if time_pref and time_pref != "Any Time":
            mins = time_pref.replace("Under ", "").replace(" min", "").strip()
            time_clause = f" ready in under {mins} minutes"

        descriptor = " ".join(parts)
        article = "an" if descriptor[0].lower() in "aeiou" else "a"
        return f"Generate {article} {descriptor}{time_clause} using these ingredients: {ing_str}"


def _build_adaptation_message(recipe_text: str, modification: str) -> str:
    return (
        f"Here is an existing recipe:\n\n{recipe_text}\n\n"
        f"Please modify it to: {modification}\n"
        f"Keep the same format (**RecipeName**, **Ingredients:**, **Instructions:**) "
        f"but adapt the recipe for the requested change."
    )


# ---------------------------------------------------------------------------
# Streaming Generation
# ---------------------------------------------------------------------------

def generate_stream(
    dish_name: str = "",
    ingredients: list[str] = None,
    diet: str = "Any Diet",
    region: str = "Any Region",
    time_pref: str = "Any Time",
):
    """
    Streaming generator: yields partial recipe text as tokens arrive.
    Logs the completed generation to MLflow after the stream finishes.
    """
    if ingredients is None:
        ingredients = []

    user_msg = _build_user_message(dish_name, ingredients, diet, region, time_pref)
    yield from _run_generation(user_msg, tag_name=dish_name or "ingredient-based")


def adapt_stream(recipe_text: str, modification: str):
    """
    Streaming generator for recipe adaptation (Make Vegan / Spicier / Quicker / custom).
    """
    user_msg = _build_adaptation_message(recipe_text, modification)
    yield from _run_generation(user_msg, tag_name=f"adapt:{modification}")


def _run_generation(user_msg: str, tag_name: str = ""):
    """Core generation loop — both generate_stream and adapt_stream call this."""
    if INFERENCE_MODE == "demo":
        yield from _demo_stream(user_msg)
        return

    # ZeroGPU path: GPU is allocated by the @spaces.GPU decorator on _run_on_gpu
    if _HAS_ZEROGPU:
        yield from _run_on_gpu(user_msg, tag_name)
    else:
        yield from _run_on_gpu_impl(user_msg, tag_name)


def _run_on_gpu_impl(user_msg: str, tag_name: str = ""):
    """
    The actual GPU inference logic. On ZeroGPU this is wrapped with @spaces.GPU
    (see below), which allocates a shared A10G GPU for the duration of the call.
    On local CUDA this runs directly.
    """
    global _tokenizer, _model

    # Lazy load on first ZeroGPU call (H200 GPU is guaranteed available here)
    if _model is None:
        _load_weights_now()

    import torch
    from transformers import TextIteratorStreamer
    from threading import Thread

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    # apply_chat_template returns BatchEncoding in transformers 5.x, raw tensor in 4.x.
    # Use tokenize=False to get the formatted string, then tokenize separately for
    # consistent behaviour across both major versions.
    prompt_str = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = _tokenizer(prompt_str, return_tensors="pt").to(_model.device)
    input_ids     = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = {
        "input_ids":          input_ids,
        "attention_mask":     attention_mask,
        "max_new_tokens":     768,
        "do_sample":          True,
        "temperature":        0.7,
        "top_p":              0.9,
        "repetition_penalty": 1.3,
        "streamer":           streamer,
    }

    t_start = time.time()
    thread = Thread(target=_model.generate, kwargs=gen_kwargs)
    thread.start()

    full_text = ""
    for new_token in streamer:
        full_text += new_token
        yield full_text

    latency_ms = int((time.time() - t_start) * 1000)
    _log_to_mlflow(user_msg, full_text, latency_ms, tag_name)


# Apply @spaces.GPU when running on HuggingFace Spaces with ZeroGPU.
# duration=120 means the GPU lease lasts up to 120 s (enough for one recipe generation).
# Locally (no `spaces` package), _run_on_gpu is just the plain function.
if _HAS_ZEROGPU:
    _run_on_gpu = _spaces_module.GPU(duration=120)(_run_on_gpu_impl)
else:
    _run_on_gpu = _run_on_gpu_impl


# ---------------------------------------------------------------------------
# MLflow Logging (LLM Observability)
# ---------------------------------------------------------------------------

def _log_to_mlflow(user_msg: str, generated_text: str, latency_ms: int, tag_name: str):
    """
    Log one inference call to MLflow Traces tab.
    Uses a nested run inside the main experiment so training runs and
    inference traces live in the same experiment, separate tabs in the UI.
    """
    try:
        with mlflow.start_run(run_name=f"inference:{tag_name[:40]}", nested=True):
            mlflow.set_tag("inference_type", "generation" if "adapt:" not in tag_name else "adaptation")
            mlflow.set_tag("dish_or_tag", tag_name[:100])
            mlflow.log_param("user_message_preview", user_msg[:200])
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("response_length_chars", len(generated_text))
            mlflow.log_metric("response_length_words", len(generated_text.split()))
    except Exception as exc:
        # Never crash the app because of a logging failure
        print(f"[mlflow] Logging failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Demo Mode — pre-written recipes for UI testing without GPU
# ---------------------------------------------------------------------------

_DEMO_RECIPES = {
    "butter chicken": """\
**Butter Chicken**

**Ingredients:**
500g Chicken (boneless, cut into pieces), 1 cup Yogurt, 2 tbsp Butter, 1 cup Tomato Puree, 1 Onion (finely chopped), 4 cloves Garlic (minced), 1 inch Ginger (grated), 1 tsp Garam Masala, 1 tsp Kashmiri Red Chilli Powder, 1 tsp Coriander Powder, 1 tsp Cumin Seeds, 3 tbsp Cream, Salt to taste, 1 tbsp Oil, Coriander Leaves to garnish

**Instructions:**
1. Marinate chicken in yogurt, red chilli powder, and salt. Set aside for at least 30 minutes.
2. Heat butter and oil in a pan. Add cumin seeds and let them splutter.
3. Add onions and sauté until golden brown, about 6-7 minutes.
4. Add ginger and garlic paste. Cook for 2 minutes until the raw aroma disappears.
5. Pour in the tomato puree. Add coriander powder, garam masala, and salt. Cook on medium heat until the oil separates, about 8-10 minutes.
6. Add the marinated chicken. Cook covered for 12-15 minutes, stirring occasionally, until the chicken is tender.
7. Stir in the cream and simmer for 3-4 minutes. Garnish with coriander leaves.
8. Serve hot with butter naan or steamed basmati rice.
""",
    "palak paneer": """\
**Palak Paneer**

**Ingredients:**
200g Paneer (cut into cubes), 3 cups Spinach (blanched and pureed), 2 Tomatoes (chopped), 1 Onion (finely chopped), 4 cloves Garlic (minced), 1 inch Ginger (grated), 1 tsp Cumin Seeds, 1 tsp Turmeric, 1 tsp Garam Masala, 1 tsp Red Chilli Powder, 2 tbsp Oil, Salt to taste, 2 tbsp Cream

**Instructions:**
1. Heat oil in a pan over medium heat. Add cumin seeds and let them splutter.
2. Add onions and sauté until golden brown, about 5-6 minutes.
3. Add ginger and garlic paste. Cook until the raw smell disappears.
4. Add chopped tomatoes, turmeric, red chilli powder, and salt. Cook until oil separates.
5. Pour in the pureed spinach. Mix well and cook for 3-4 minutes on medium heat.
6. Add paneer cubes and garam masala. Stir gently and cook for 2-3 minutes.
7. Finish with cream, stir through, and serve hot with roti or rice.
""",
    "dal makhani": """\
**Dal Makhani**

**Ingredients:**
1 cup Urad Dal (whole black lentils), 3 tbsp Rajma (kidney beans), 2 Tomatoes (pureed), 1 Onion (finely chopped), 4 cloves Garlic, 1 inch Ginger, 2 tbsp Butter, 3 tbsp Cream, 1 tsp Cumin Seeds, 1 tsp Red Chilli Powder, 1 tsp Garam Masala, Salt to taste

**Instructions:**
1. Soak urad dal and rajma overnight. Pressure cook with salt until completely soft.
2. Mash about half the cooked dal with the back of a spoon for a creamy texture.
3. Heat butter in a pan. Add cumin seeds, then onions. Sauté until deep golden.
4. Add ginger-garlic paste, then tomato puree and red chilli powder. Cook until oil separates.
5. Add the cooked dal to the masala. Simmer on very low heat for 20-25 minutes, stirring occasionally.
6. Finish with cream and garam masala. Serve with naan or roti.
""",
    "aloo gobi": """\
**Aloo Gobi**

**Ingredients:**
2 Potatoes (peeled and cubed), 1 small Cauliflower (cut into florets), 1 Onion (sliced), 2 Tomatoes (chopped), 1 tsp Cumin Seeds, 1 tsp Turmeric, 1 tsp Coriander Powder, 1 tsp Garam Masala, 1/2 tsp Red Chilli Powder, 2 tbsp Oil, Salt to taste, Coriander Leaves to garnish

**Instructions:**
1. Heat oil in a kadai. Add cumin seeds and let them crackle.
2. Add onions and sauté until translucent. Add tomatoes and cook until soft.
3. Add turmeric, coriander powder, red chilli powder, and salt. Mix well.
4. Add potatoes and cauliflower florets. Toss to coat with the masala.
5. Cover and cook on low heat for 15-18 minutes, stirring occasionally, until vegetables are tender.
6. Add garam masala and garnish with coriander leaves. Serve hot with roti.
""",
}

_DEFAULT_DEMO = """\
**{dish} — Demo Recipe**

**Ingredients:**
Paneer (200g), Spinach (3 cups), Tomato (2 chopped), Onion (1 finely chopped), Garlic (4 cloves), Ginger (1 inch), Cumin Seeds (1 tsp), Turmeric (1 tsp), Garam Masala (1 tsp), Oil (2 tbsp), Salt to taste

**Instructions:**
1. Heat oil in a pan. Add cumin seeds and let them splutter over medium heat.
2. Add onions and sauté until golden brown, about 5 minutes. Add ginger and garlic.
3. Add tomatoes and all dry spices. Cook until the oil separates from the masala.
4. Add the main ingredients. Cook covered for 12-15 minutes until tender.
5. Adjust seasoning and garnish with fresh coriander. Serve hot.

(Demo mode — real model would generate an authentic {dish} recipe. Run with GPU for actual output.)
"""


def _demo_stream(user_msg: str):
    """
    Simulate streaming with a recipe matched to the user's request.
    Looks up a pre-written recipe by dish name keyword, falls back to a generic template.
    """
    # Extract dish name from user message for lookup
    msg_lower = user_msg.lower()
    recipe = None
    for key, text in _DEMO_RECIPES.items():
        if key in msg_lower:
            recipe = text
            break

    if recipe is None:
        # Try to extract a dish name from the message for the template
        import re
        name_match = re.search(r"recipe for (.+?)$", user_msg, re.IGNORECASE)
        dish = name_match.group(1).strip().title() if name_match else "Indian Dish"
        recipe = _DEFAULT_DEMO.format(dish=dish)

    accumulated = ""
    for char in recipe:
        accumulated += char
        if char in (" ", "\n", ",", "."):
            time.sleep(0.012)
            yield accumulated
    yield accumulated
    _log_to_mlflow(user_msg, recipe, latency_ms=0, tag_name="demo")
