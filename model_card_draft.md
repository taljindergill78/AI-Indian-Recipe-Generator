---
language:
- en
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- text-generation
- fine-tuned
- qlora
- lora
- peft
- indian-cuisine
- recipe-generation
- food
datasets:
- Anupam007/indian-recipe-dataset
pipeline_tag: text-generation
---

# Indian Recipe Generator — LLaMA 3.2-3B QLoRA Fine-Tune

A QLoRA fine-tuned version of [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
trained on 3,263 authentic Indian recipes. Given a dish name, diet type, and regional
cuisine, the model generates a structured recipe with a full ingredients list and
step-by-step cooking instructions.

---

## Model Description

The base LLaMA 3.2-3B-Instruct model has general language understanding but no
domain-specific knowledge of Indian cuisine — it hallucinates ingredients, misses
regional cooking techniques, and produces generic Western-style recipe formats. This
fine-tune teaches the model:

- **Ingredient vocabulary**: turmeric, asafoetida, methi, kasuri methi, hing, and 200+
  other ingredients common in Indian cooking
- **Regional variation**: North Indian, South Indian, Bengali, Gujarati, Rajasthani,
  and other regional cuisines each have distinct flavor profiles and techniques
- **Diet-aware generation**: Vegetarian, Non-Vegetarian, and Vegan recipe variants
- **Structured output format**: bold-header format with `**Ingredients:**` and
  `**Instructions:**` sections, matching the training data format

---

## Intended Use

**Intended for:**
- Generating authentic Indian recipes from a dish name + diet + region prompt
- Portfolio demonstration of end-to-end LLM fine-tuning with QLoRA

**Not intended for:**
- Medical or dietary advice
- Production food safety applications
- Real-time serving without GPU (CPU inference is very slow for a 3B model)

---

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Step 1: Load base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# Step 2: Load the LoRA adapter on top
model = PeftModel.from_pretrained(base_model, "taljindergill78/indian-recipe-llama3.2-qlora")
model.eval()

# Step 3: Load the tokenizer (stored alongside adapter for convenience)
tokenizer = AutoTokenizer.from_pretrained("taljindergill78/indian-recipe-llama3.2-qlora")

# Step 4: Generate a recipe
def generate_recipe(dish_name, diet="Vegetarian", region="North Indian"):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Indian chef. Generate authentic Indian recipes "
                "with detailed ingredients and clear step-by-step cooking instructions."
            ),
        },
        {
            "role": "user",
            "content": f"Generate a {diet} {region} recipe for {dish_name}",
        },
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,  # prevents ingredient repetition loops
        )

    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Example
print(generate_recipe("Dal Makhani"))
print(generate_recipe("Dosa", diet="Vegetarian", region="South Indian"))
print(generate_recipe("Chicken Biryani", diet="Non Vegetarian", region="Hyderabadi"))
```

**Hardware requirements:** GPU with ≥8GB VRAM recommended (A100 / T4 / RTX 3080+).
Loads as ~3GB with 4-bit quantization.

---

## Training Data

Fine-tuned on **3,263 Indian recipes** from the
[Anupam007/indian-recipe-dataset](https://huggingface.co/datasets/Anupam007/indian-recipe-dataset)
(originally scraped from archanaskitchen.com).

**Dataset split used for training:**
- Train: 3,263 recipes
- Validation: 250 recipes (used for per-epoch eval_loss during training)
- Test: 500 recipes (held out; used for final evaluation metrics below)

**Filtering applied:** The raw dataset contains ~5,938 rows including Continental, Italian,
and other non-Indian cuisines. Only rows tagged as Indian cuisine were retained.

**Prompt format:** Each training example uses the chat template format:
```
System: You are an expert Indian chef...
User: Generate a {diet} {region} recipe for {dish_name}
Assistant: **{recipe_name}**\n\n**Ingredients:**\n...\n\n**Instructions:**\n...
```
Loss was computed only on the assistant response tokens (loss masking via TRL's
`assistant_only_loss=True`), so the model learns to generate recipes, not to repeat prompts.

---

## Training Procedure

**Method:** QLoRA — 4-bit NF4 quantization of the base model (frozen) + LoRA adapters
on the 7 projection layers of each transformer block

| Parameter | Value | Reason |
|---|---|---|
| LoRA rank (r) | 16 | Community default for 3B models; broader than r=8 for domain adaptation |
| LoRA alpha | 32 | Scaling factor = alpha/r = 2.0 |
| LoRA target modules | q, k, v, o, gate, up, down proj | All 7 projection layers per block |
| Trainable parameters | 24,313,856 / 3,237,063,680 | **0.75%** of total model params |
| Epochs | 3 | Converged by epoch 3; no overfitting observed |
| Batch size | 4 (physical) × 4 (grad accum) = **16 effective** | A100-80GB headroom |
| Learning rate | 2e-4 with cosine decay | Standard for QLoRA fine-tuning |
| LR warmup | 5% of steps (first 30 steps) | Prevents unstable early updates |
| Max sequence length | 1024 tokens | Covers 95%+ of recipe lengths |
| Optimizer | AdamW | No paging needed — 80GB VRAM has headroom |
| Precision | BF16 | LLaMA 3.2's native dtype; A100 has native BF16 Tensor Cores |

**Hardware:** NVIDIA A100-SXM4-80GB (RunPod Community Cloud)
**Training time:** 28 minutes 16 seconds (612 steps)
**VRAM peak:** 41 GB / 80 GB (52%)

---

## Evaluation Results

All metrics computed on the full **500-row held-out test set** (not seen during training or
validation). Baselines use greedy decoding. Fine-tuned model uses nucleus sampling
(`do_sample=True`, `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.3`,
`max_new_tokens=768`). 95% bootstrap confidence intervals reported where available.

### Before vs After Fine-Tuning

| Model | Ingredient F1 | ROUGE-L | BERTScore F1 | BLEU |
|---|---|---|---|---|
| Phi-3-mini-4k (baseline) | 0.0188 | 0.0984 | 0.8018 | 0.0048 |
| LLaMA 3.2-1B (baseline) | 0.0255 | 0.1711 | 0.8372 | 0.0143 |
| LLaMA 3.2-3B (baseline) | 0.0308 | 0.1954 | 0.8455 | 0.0195 |
| **LLaMA 3.2-3B fine-tuned** (this model) | **0.0992 ↑3.2×** | 0.1835 | **0.8514 ↑** | **0.0262 ↑34%** |

**Fine-tuned model 95% CIs:** Ingredient F1 [0.0924, 0.1068] · ROUGE-L [0.1802, 0.1872] · BERTScore [0.8503, 0.8524]

> **Note on metric behavior:**
>
> **Ingredient F1 (3.2× improvement)** is the primary signal that fine-tuning worked. The model
> learned Indian ingredient vocabulary — turmeric, asafoetida, methi, kasuri methi, poppy seeds,
> and 200+ other region-specific ingredients the base model had no training signal on.
>
> **ROUGE-L is slightly lower than baseline** (0.1835 vs 0.1954). This is the expected behavior
> for a fine-tuned generative model: the fine-tuned model generates plausible, authentic recipes
> that differ in wording from the specific reference text in the test set. ROUGE-L measures exact
> text overlap (longest common subsequence) — it penalizes creativity. A model that perfectly
> copies the training data would score higher on ROUGE-L but be useless.
>
> **BERTScore improved (+0.7%)**, confirming the generated instructions are semantically more
> appropriate even when the exact phrasing differs. BERTScore going up while ROUGE-L goes down
> is the healthy pattern for a creative generative model.
>
> **BLEU improved 34%** — solid n-gram overlap improvement on cooking instructions.

### Training Convergence (Validation Set, 250 recipes)

| Epoch | eval_loss | Token Accuracy | Train/Eval Gap |
|---|---|---|---|
| 1 | 1.302 | 66.38% | 0.001 (no overfitting) |
| 2 | 1.250 | 67.43% | 0.078 (small, healthy) |
| 3 | **1.249** | **67.55%** | 0.139 (small, healthy) |

Best checkpoint: Epoch 3 (selected automatically by `load_best_model_at_end=True`).

### Metric Definitions

- **Ingredient F1**: Token-level overlap between generated and reference ingredient lists
  (after ingredient name normalization with rapidfuzz). Measures whether the model
  generates the correct ingredients.
- **ROUGE-L**: Longest common subsequence overlap between generated and reference
  instructions. Measures structural similarity of cooking steps.
- **BERTScore F1**: Semantic similarity of instructions using RoBERTa embeddings.
  Measures whether the generated instructions *mean* the same thing even if worded
  differently.
- **BLEU**: N-gram precision of generated instructions against reference. Strict
  surface-form match — expected to be low for generative recipes.

---

## Limitations

- **Vocabulary bias**: Training data is from a single source (archanaskitchen.com).
  Less common regional dishes (Northeastern Indian, tribal cuisines) are
  underrepresented.
- **Quantity accuracy**: Ingredient quantities may not always be correct for the
  number of servings generated.
- **Hallucination**: The model may occasionally generate plausible-sounding but
  incorrect steps for dishes it saw rarely in training.
- **Language**: English only. The training data is English-translated recipes.
- **Format dependency**: The model expects the exact system prompt and user message
  format shown in the usage example. Deviating from it may produce off-format outputs.

---

## Repository

Training code, evaluation scripts, and full documentation:
[github.com/taljindergill78/AI-Indian-Recipe-Generator](https://github.com/taljindergill78)

Built as part of an end-to-end LLM fine-tuning portfolio project.
MS Data Science, Arizona State University.
