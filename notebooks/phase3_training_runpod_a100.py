# %% [markdown]
# # Phase 3: QLoRA Fine-Tuning — RunPod A100 Training Run
#
# **Project**: AI Indian Recipe Generator
# **Date**: 2026-05-19
# **Purpose**: Documents the exact setup, environment, and commands used to fine-tune
#              LLaMA 3.2-3B-Instruct on 3,263 Indian recipes using QLoRA on a RunPod A100.
#
# This notebook **replaces** `phase3_training_kaggle.py` for the actual production training run.
# The Kaggle notebook was used for earlier experiments and debugging (Phase 3, Steps 2 & 3).
# This RunPod run is the final, successful fine-tuning that produced the published adapter.
#
# **Result**: https://huggingface.co/taljindergill78/indian-recipe-llama3.2-qlora

# %% [markdown]
# ## 1. Hardware Selection — Why A100 and Not Kaggle/Colab
#
# The T4 GPU on Kaggle was unusable for this training because:
# - T4 has NO native BF16 Tensor Cores (falls back to FP32 at 8 TFLOPS instead of 65 TFLOPS FP16)
# - LLaMA 3.2-3B is released in BF16 natively — training it on T4 hit 112 sec/step
# - A100 SXM4 has native BF16 Tensor Cores at 312 TFLOPS and HBM2 bandwidth at 2,000 GB/s
#
# **Pod selected on RunPod Community Cloud:**
# - GPU:       A100-SXM4-80GB (80 GB VRAM)
# - CPU:       AMD EPYC 7742 (16 vCPU)
# - RAM:       232 GB system RAM
# - Container disk: 30 GB
# - Volume:    20 GB at /workspace (persistent — survives pod stop)
# - Template:  RunPod PyTorch (CUDA 12.4.1, Python 3.11)
# - Cost:      $1.50/hr on-demand
# - Total run: ~1 hour end-to-end (setup + training + push)
#
# **Telemetry during training:**
# - VRAM usage:       41 GB / 80 GB (52%)
# - GPU utilization:  92%
# - Power:            396W / 400W (P0 state — full performance)
# - Temperature:      63°C (max 80°C before throttle)
# - Training speed:   ~2.7 sec/step

# %% [markdown]
# ## 2. Environment Setup
#
# The RunPod PyTorch template ships with:
# - PyTorch 2.4.1+cu124
# - Python 3.11
# - CUDA 12.4.1
#
# PyTorch 2.4.1 is incompatible with transformers 5.x because transformers added MoE
# (Mixture of Experts) CUDA registration code that uses a PyTorch 2.5+ API.
# Solution: upgrade PyTorch to 2.5.1 first, then install the rest.

# %% [markdown]
# ### Fix: blinker conflict (distutils-installed package)
# The RunPod image ships blinker 1.4 installed via apt (not pip).
# pip refuses to uninstall it. Fix: reinstall with --ignore-installed.

# %%
# TERMINAL COMMAND (run in JupyterLab terminal, not as Python):
# pip install -q blinker --ignore-installed

# %% [markdown]
# ### Fix: PyTorch 2.4.1 → 2.5.1 (required for transformers 5.x MoE import)

# %%
# TERMINAL COMMAND:
# pip install -q "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124

# %% [markdown]
# ### Install training dependencies

# %%
# TERMINAL COMMAND:
# pip install -q transformers>=4.45.0 trl>=1.0.0 peft>=0.13.0 bitsandbytes>=0.46.0 \
#             accelerate>=0.34.0 datasets mlflow huggingface_hub nltk

# %% [markdown]
# ### Verify versions

# %%
import torch
import transformers
import trl
import peft
import bitsandbytes

print(f"torch:          {torch.__version__}")      # 2.5.1+cu124
print(f"transformers:   {transformers.__version__}")  # 5.8.1
print(f"trl:            {trl.__version__}")        # 1.4.0
print(f"peft:           {peft.__version__}")       # 0.13.x
print(f"cuda available: {torch.cuda.is_available()}")
print(f"GPU:            {torch.cuda.get_device_name(0)}")  # NVIDIA A100-SXM4-80GB

# %% [markdown]
# ## 3. File Setup
#
# The training zip (`ai-indian-recipe-train.zip`) was built locally on Mac and uploaded
# via the JupyterLab file browser (upload button in the left panel).
#
# Contents of the zip:
# - data/processed/train.csv       (3,263 training recipes)
# - data/processed/val.csv         (250 validation recipes)
# - models/train_llama3.py         (training script)
# - models/base_utils/common_utils.py
# - models/__init__.py
# - params.yaml                    (hyperparameters and config)

# %%
# TERMINAL COMMAND (after uploading zip via JupyterLab UI):
# python -c "import zipfile; zipfile.ZipFile('ai-indian-recipe-train.zip').extractall('.')"
# (unzip is not installed on the RunPod image — use Python's zipfile module)

# %% [markdown]
# ## 4. Training Configuration
#
# All hyperparameters are defined in `models/train_llama3.py` under `TRAIN_CONFIG`.
# Here are the key values and why each was chosen for A100:

# %%
# Key training configuration (from models/train_llama3.py TRAIN_CONFIG):
TRAIN_CONFIG_REFERENCE = {
    # ---- LoRA ----
    "lora_r": 16,                        # LoRA rank — width of A and B matrices
    "lora_alpha": 32,                    # scaling = alpha/r = 2.0
    "lora_target_modules": [             # all 7 projection layers in each block
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,

    # ---- Training ----
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,    # A100 can fit 2x what T4 could
    "gradient_accumulation_steps": 4,   # effective batch = 4 x 4 = 16
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",       # warmup → peak → cosine decay to ~0
    "warmup_ratio": 0.05,                # first 5% of steps ramp LR up
    "max_length": 1024,                  # max tokens per training example

    # ---- A100-specific choices ----
    "optim": "adamw_torch",              # no paging needed — 80GB VRAM has headroom
    "gradient_checkpointing": False,     # A100 can hold full activations (no recompute)
    "bf16": True,                        # LLaMA 3.2 native dtype; A100 has BF16 Tensor Cores
    "fp16": False,                       # FP16 GradScaler crashes on native BF16 tensors

    # ---- Checkpointing ----
    "eval_strategy": "epoch",            # validate on 250 recipes after each epoch
    "save_strategy": "epoch",
    "load_best_model_at_end": True,      # auto-select best checkpoint by eval_loss
    "metric_for_best_model": "eval_loss",
    "save_total_limit": 2,
}

# Quantization config (4-bit NF4)
QUANT_CONFIG_REFERENCE = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",        # NormalFloat4 — designed for neural net weight distributions
    "bnb_4bit_compute_dtype": "bfloat16",  # dequantize to BF16 for matmul
    "bnb_4bit_use_double_quant": True,   # quantize the scale factors too (~0.4GB saved)
}

print("Training config loaded. Trainable params: 24,313,856 / 3,237,063,680 (0.75%)")

# %% [markdown]
# ## 5. Run Training
#
# Set environment variables and launch. The script downloads the base model from HuggingFace
# (~6.4GB, ~15 seconds at A100 pod speeds), tokenizes the dataset, and begins training.

# %%
# TERMINAL COMMANDS:
# export HF_TOKEN=hf_your_read_token_here
# export MODEL_OUTPUT_DIR=/workspace/models/trained
# export MLFLOW_TRACKING_URI=/workspace/mlruns
# python models/train_llama3.py

# %% [markdown]
# ## 6. Training Results
#
# **Total time**: 1,696 seconds = 28 minutes 16 seconds
# **Total steps**: 612 (204 steps/epoch × 3 epochs)
# **Speed**: ~2.7 seconds/step

# %%
# Epoch-by-epoch validation results:
results = {
    "epoch_1": {
        "eval_loss": 1.302,
        "eval_accuracy": 0.6638,
        "eval_entropy": 1.313,
        "train_loss_at_end": 1.301,
        "train_eval_gap": 0.001,  # essentially zero — no overfitting
    },
    "epoch_2": {
        "eval_loss": 1.250,
        "eval_accuracy": 0.6743,
        "eval_entropy": 1.232,
        "train_loss_at_end": 1.172,
        "train_eval_gap": 0.078,  # small, healthy
    },
    "epoch_3": {
        "eval_loss": 1.249,
        "eval_accuracy": 0.6755,
        "eval_entropy": 1.171,
        "train_loss_at_end": 1.110,
        "train_eval_gap": 0.139,  # small, healthy — no overfitting
    },
}

# Best checkpoint: epoch 3 (eval_loss 1.249, marginally better than epoch 2's 1.250)
# load_best_model_at_end=True selected it automatically

for epoch, metrics in results.items():
    print(f"{epoch}: eval_loss={metrics['eval_loss']}, "
          f"accuracy={metrics['eval_accuracy']:.1%}, "
          f"gap={metrics['train_eval_gap']}")

# %% [markdown]
# ## 7. Push Adapter to HuggingFace Hub
#
# The initial push via `trainer.push_to_hub()` failed with 403 because `HF_TOKEN` was
# set to a read-only token. Fix:
# 1. Create a WRITE token at huggingface.co → Settings → Access Tokens
# 2. Unset the read token env var
# 3. Use the new `hf` CLI (replaces deprecated `huggingface-cli`)

# %%
# TERMINAL COMMANDS (after getting write token):
# unset HF_TOKEN
# hf auth login --token hf_your_write_token_here
# hf upload taljindergill78/indian-recipe-llama3.2-qlora /workspace/models/trained/final .

# Result: 97.3 MB uploaded in ~15 seconds
# URL: https://huggingface.co/taljindergill78/indian-recipe-llama3.2-qlora

# %% [markdown]
# ## 8. What Was Uploaded to HuggingFace Hub
#
# Only the LoRA ADAPTER is on the Hub — NOT the 6.4GB base model.
#
# Files:
# - adapter_model.safetensors  (97 MB) — LoRA A/B matrices for 7 target layers
# - tokenizer.json             (17 MB) — same vocab as base, stored for convenience
# - training_args.bin          (5 KB)  — training config snapshot
#
# To load and use the fine-tuned model:

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

def load_finetuned_model():
    """Load the fine-tuned Indian recipe model from HuggingFace Hub."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model (from Meta's repo)
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    # Layer our adapter on top
    model = PeftModel.from_pretrained(
        base_model,
        "taljindergill78/indian-recipe-llama3.2-qlora"
    )

    # Tokenizer stored alongside adapter for convenience
    tokenizer = AutoTokenizer.from_pretrained(
        "taljindergill78/indian-recipe-llama3.2-qlora"
    )

    return model, tokenizer


def generate_recipe(model, tokenizer, dish_name, diet="Vegetarian", region="North Indian"):
    """Generate a recipe using the fine-tuned model."""
    messages = [
        {
            "role": "system",
            "content": "You are an expert Indian chef. Generate authentic Indian recipes "
                       "with detailed ingredients and clear step-by-step cooking instructions."
        },
        {
            "role": "user",
            "content": f"Generate a {diet} {region} recipe for {dish_name}"
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
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Strip input tokens from output
    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Usage:
# model, tokenizer = load_finetuned_model()
# recipe = generate_recipe(model, tokenizer, "Dal Makhani")
# print(recipe)

# %% [markdown]
# ## 9. Stopping the Pod
#
# Once the adapter is confirmed on HuggingFace Hub, stop the pod:
# RunPod dashboard → your pod → Stop
#
# What is preserved:
# - LoRA adapter: on HuggingFace Hub ✓
# - Training script + data: in GitHub repo ✓
# - Base model: re-downloadable from Meta ✓
# - MLflow logs: LOST (local to container) — Phase 4 note: use remote MLflow or export first
#
# What is lost on pod stop:
# - Downloaded base model weights (6.4GB on container disk)
# - Python environment (will need pip install again if pod restarted)
# - mlruns/ directory

# %% [markdown]
# ## 10. Next Steps — Phase 4 (Serving)
#
# The adapter is on HuggingFace Hub. Next:
# 1. Write a model card (README on the Hub repo) with eval results and usage instructions
# 2. Build a Gradio inference app
# 3. Deploy to HuggingFace Spaces
# 4. Add multi-turn adaptation (Make Vegan / Spicier / Quicker)
