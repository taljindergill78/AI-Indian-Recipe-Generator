"""
Phase 3 Training — Kaggle Notebook Reference Script
=====================================================
HOW TO USE THIS FILE:
  Each section below corresponds to ONE Kaggle notebook cell.
  The code between the cell header and the next cell header is what you paste.
  Lines starting with '#' are explanation comments — they are NOT code to paste.
  Lines NOT starting with '#' are the actual code to run in that cell.

Kaggle notebook settings:
  - Accelerator: GPU T4 x2  (P100 will FAIL: compute 6.0, bitsandbytes requires 7.5+)
  - Internet: ON             (needed for pip install and HuggingFace model download)
  - Persistence: Files       (so output survives the session)

Dataset to attach BEFORE running:
  Name: ai-indian-recipe-train
  Required files inside it:
    models/train_llama3.py
    models/base_utils/common_utils.py
    data/processed/train.csv
    data/processed/val.csv
    params.yaml

HuggingFace token:
  Add HF_TOKEN ONCE at kaggle.com → Your Account → Settings → Add-ons: Secrets
  It works in every notebook — you don't need to re-add per notebook.
"""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1  —  Install dependencies                                         ║
# ║  Paste everything below this box, up to the next box, into Cell 1        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Kaggle T4 image has PyTorch pre-installed. Only add the HF + TRL stack.
# Pin versions to match what was tested locally.

!pip install -q \
    "transformers>=4.45" \
    "peft>=0.13" \
    "trl>=1.0" \
    "bitsandbytes>=0.46" \
    "datasets>=2.20" \
    "mlflow>=2.15" \
    "accelerate>=0.30"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2  —  HuggingFace authentication                                   ║
# ║  Paste everything below this box, up to the next box, into Cell 2        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# HF_TOKEN is an account-level Kaggle secret — added once, works in every notebook.
# This sets HF_TOKEN as an environment variable so train_llama3.py can read it.

import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3  —  Path setup                                                   ║
# ║  Paste everything below this box, up to the next box, into Cell 3        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Kaggle dataset mounts are READ-ONLY.
# Strategy: copy the entire dataset to /kaggle/working/ (writable), then chdir there.
# This mirrors how Phase 2 evaluation was run.
#
# First run: print(os.listdir('/kaggle/input/')) to confirm the exact mount name.
# The DATASET_DIR path below assumes the dataset is mounted as shown — verify before running.

import os, sys, shutil

DATASET_DIR = "/kaggle/input/datasets/taljindersingh/ai-indian-recipe-train"

shutil.copytree(DATASET_DIR, '/kaggle/working/', dirs_exist_ok=True)
os.chdir('/kaggle/working/')
sys.path.insert(0, '/kaggle/working/')

# Redirect model output to writable absolute path.
# train_llama3.py reads MODEL_OUTPUT_DIR env var if set, else falls back to params.yaml.
os.environ["MODEL_OUTPUT_DIR"] = "/kaggle/working/models/trained/"

print("Working directory:", os.getcwd())
print("Contents:", sorted(os.listdir('.')))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4  —  Verify GPU                                                   ║
# ║  Paste everything below this box, up to the next box, into Cell 4        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Expected output:
#   CUDA: True
#   GPU:  Tesla T4
#   VRAM: 15.84 GB
#
# If you see P100 here: STOP. Switch to T4 x2 in notebook settings.
# P100 has compute capability 6.0 — bitsandbytes 4-bit quantization requires 7.5+.

import torch
print("CUDA:", torch.cuda.is_available())
print("GPU: ", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5  —  Dry run (config check, no model loading)                     ║
# ║  Paste everything below this box, up to the next box, into Cell 5        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Always run this before full training to catch config errors early.
# Takes ~5 seconds. Verifies params.yaml is readable and all paths are correct.
# Expected output:
#   === DRY RUN — config only ===
#     Model:        meta-llama/Llama-3.2-3B-Instruct
#     Train data:   data/processed/train.csv
#     Val data:     data/processed/val.csv
#     Output dir:   /kaggle/working/models/trained/
#     LoRA rank:    16  alpha: 32  ...

!python models/train_llama3.py --dry-run


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6  —  Full training run                                            ║
# ║  Paste everything below this box, up to the next box, into Cell 6        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Expected timeline on T4 x2 (using only cuda:0, ~16GB VRAM):
#   Package install:    ~2 min
#   Model download:     ~5 min  (LLaMA 3.2-3B-Instruct, ~6GB from HuggingFace Hub)
#   Epoch 1 training:   ~25-35 min  (train loss: ~2.5 → ~1.5)
#   Epoch 1 eval:       ~3-5 min
#   Epoch 2 training:   ~25-35 min  (train loss: ~1.2 → ~1.0)
#   Epoch 2 eval:       ~3-5 min
#   Epoch 3 training:   ~25-35 min  (train loss: ~0.9 → ~0.8)
#   Epoch 3 eval:       ~3-5 min
#   Hub push:           ~2 min
#   Total:              ~1.5-1.8 hours  (well within 9-hour session limit)
#
# MLflow logs to /kaggle/working/mlruns/ automatically (report_to="mlflow" in TRAIN_CONFIG).
# Download mlruns/ from the Kaggle Output tab after training.
#
# To test without Hub push:  !python models/train_llama3.py --no-push
# To override learning rate: !python models/train_llama3.py --lr 1e-4
# To override LoRA rank:     !python models/train_llama3.py --lora-r 32

!python models/train_llama3.py


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7  —  Verify model was saved                                       ║
# ║  Paste everything below this box, up to the next box, into Cell 7        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Expected files: adapter_config.json, adapter_model.safetensors (~80MB),
#                 tokenizer.json, tokenizer_config.json, special_tokens_map.json

import os
final_dir = "/kaggle/working/models/trained/final/"
print("Final model directory exists:", os.path.isdir(final_dir))
if os.path.isdir(final_dir):
    print("Files saved:")
    for f in sorted(os.listdir(final_dir)):
        size_mb = os.path.getsize(os.path.join(final_dir, f)) / 1e6
        print(f"  {f}  ({size_mb:.1f} MB)")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8  —  View MLflow training summary                                 ║
# ║  Paste everything below this box, up to the next box, into Cell 8        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Shows all metrics and params logged during training.
# After this cell, download /kaggle/working/mlruns/ from the Kaggle Output tab
# to view full training curves locally with: mlflow ui

import mlflow
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("qlora-indian-recipe-finetune")
if exp:
    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
    if runs:
        run = runs[0]
        print("Run ID:", run.info.run_id)
        print("Status:", run.info.status)
        print("\nMetrics logged:")
        for k, v in sorted(run.data.metrics.items()):
            print(f"  {k}: {v:.4f}")
        print("\nParams logged:")
        for k, v in sorted(run.data.params.items()):
            print(f"  {k}: {v}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9  —  Quick inference sanity check                                 ║
# ║  Paste everything below this box into Cell 9                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Load the fine-tuned adapter and generate one recipe.
# Signs of successful fine-tuning:
#   - Uses exact ingredient names from training data (not generic "black lentils")
#   - Follows **bold header** format: **RecipeName**, **Ingredients:**, **Instructions:**
#   - Instructions are numbered step-by-step
#   - Total output length ~200-400 tokens (full recipe)

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "/kaggle/working/models/trained/final/"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
ft_model = PeftModel.from_pretrained(base_model, adapter_path)
ft_model.eval()

messages = [
    {"role": "system",
     "content": "You are an expert Indian chef. Generate authentic Indian recipes with "
                "detailed ingredients and clear step-by-step cooking instructions."},
    {"role": "user",
     "content": "Generate a Vegetarian North Indian main course recipe for Dal Makhani"},
]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

with torch.inference_mode():
    output_ids = ft_model.generate(
        input_ids, max_new_tokens=512, temperature=0.7, do_sample=True
    )
generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
print(generated)
