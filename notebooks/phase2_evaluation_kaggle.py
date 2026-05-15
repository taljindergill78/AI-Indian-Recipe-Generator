"""
Phase 2 Evaluation — Kaggle Notebook Reference Script
======================================================
HOW TO USE THIS FILE:
  Each section below corresponds to ONE Kaggle notebook cell.
  The code between the cell header and the next cell header is what you paste.
  Lines starting with '#' are explanation comments — they are NOT code to paste.
  Lines NOT starting with '#' are the actual code to run in that cell.

Kaggle notebook settings:
  - Accelerator: GPU T4 x2  (T4 or P100 both work for evaluation — no 4-bit quant needed,
    but T4 is preferred since it also supports bf16)
  - Internet: ON             (needed for pip install and HuggingFace model download)
  - Persistence: Files       (so output survives the session)

Dataset to attach BEFORE running:
  Name: ai-indian-recipe-eval
  Required files inside it:
    scripts/evaluation/evaluate.py
    scripts/evaluation/metrics.py
    scripts/evaluation/parser.py
    scripts/__init__.py
    scripts/evaluation/__init__.py
    models/base_utils/common_utils.py
    data/processed/test.csv
    params.yaml

HuggingFace token:
  Add HF_TOKEN ONCE at kaggle.com → Your Account → Settings → Add-ons: Secrets
  It works in every notebook — you don't need to re-add per notebook.
"""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1  —  Install dependencies                                         ║
# ║  Paste everything below this box, up to the next box, into Cell 1        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Kaggle T4 image has PyTorch pre-installed. Add evaluation + HuggingFace stack.

!pip install -q \
    "transformers>=4.45" \
    "bitsandbytes>=0.46" \
    "datasets>=2.20" \
    "mlflow>=2.15" \
    "accelerate>=0.30" \
    "bert-score>=0.3.13" \
    "sacrebleu>=2.3" \
    "rouge-score>=0.1.2" \
    "rapidfuzz>=3.6"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2  —  HuggingFace authentication                                   ║
# ║  Paste everything below this box, up to the next box, into Cell 2        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# HF_TOKEN is an account-level Kaggle secret — added once, works in every notebook.

import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3  —  Path setup                                                   ║
# ║  Paste everything below this box, up to the next box, into Cell 3        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Kaggle dataset mounts are READ-ONLY.
# Copy entire dataset to /kaggle/working/ (writable), then chdir there.
#
# First run: print(os.listdir('/kaggle/input/')) to confirm the exact mount name.

import os, sys, shutil

DATASET_DIR = "/kaggle/input/datasets/taljindersingh/ai-indian-recipe-eval"

shutil.copytree(DATASET_DIR, '/kaggle/working/', dirs_exist_ok=True)
os.chdir('/kaggle/working/')
sys.path.insert(0, '/kaggle/working/')

print("Working directory:", os.getcwd())
print("Contents:", sorted(os.listdir('.')))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4  —  Verify GPU                                                   ║
# ║  Paste everything below this box, up to the next box, into Cell 4        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Both T4 and P100 work for evaluation (no 4-bit quant required here).
# Expected output: CUDA: True, GPU: Tesla T4, VRAM: ~15.84 GB

import torch
print("CUDA:", torch.cuda.is_available())
print("GPU: ", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5  —  Run evaluation for LLaMA 3.2-1B-Instruct (Candidate 1)      ║
# ║  Paste everything below this box, up to the next box, into Cell 5        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# evaluate.py reads CONFIG["model_key"] to pick the model.
# Edit params.yaml or pass model_key via env var to switch candidates.
# Expected runtime: ~30-45 min for 500 test rows on T4.
# Outputs saved to: data/processed/llama3_1b_baseline_prompts.json
#                   data/processed/llama3_1b_baseline_metrics.json
#
# To evaluate LLaMA 3B: change model_key to "llama3_3b" in scripts/evaluation/evaluate.py
# To evaluate Phi-3:    change model_key to "phi3_mini"

!python scripts/evaluation/evaluate.py


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6  —  View results summary                                         ║
# ║  Paste everything below this box, up to the next box, into Cell 6        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Read the saved metrics JSON and print the summary table.
# Look for: ingredient_f1, rouge_l, bleu_4, bertscore_f1 with 95% CI bounds.

import json, glob

metrics_files = sorted(glob.glob("data/processed/*_metrics.json"))
for path in metrics_files:
    print(f"\n{'='*60}")
    print(f"Results: {path}")
    print('='*60)
    with open(path) as f:
        metrics = json.load(f)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    {sub_k}: {sub_v:.4f}" if isinstance(sub_v, float) else f"    {sub_k}: {sub_v}")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7  —  View MLflow experiment logs                                  ║
# ║  Paste everything below this box into Cell 7                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# After this cell, download /kaggle/working/mlruns/ from the Kaggle Output tab
# to view locally with: mlflow ui

import mlflow
client = mlflow.tracking.MlflowClient()

for exp_name in ["candidate-eval-llama1b", "candidate-eval-llama3b", "candidate-eval-phi3mini"]:
    exp = client.get_experiment_by_name(exp_name)
    if exp:
        runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
        if runs:
            run = runs[0]
            print(f"\n{'='*50}")
            print(f"Experiment: {exp_name}")
            print(f"Run ID: {run.info.run_id}  Status: {run.info.status}")
            print("Metrics:")
            for k, v in sorted(run.data.metrics.items()):
                print(f"  {k}: {v:.4f}")
