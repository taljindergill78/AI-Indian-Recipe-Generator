"""
Phase 3 Fine-Tuned Model Evaluation
=====================================
Runs the fine-tuned LLaMA 3.2-3B LoRA adapter through all 500 test rows
and computes the same metrics as Phase 2 evaluate.py (Ingredient F1, ROUGE-L,
BERTScore, BLEU) so results are directly comparable.

Usage (on RunPod JupyterLab terminal):
    export HF_TOKEN=hf_your_read_token_here
    python scripts/evaluation/evaluate_finetuned.py

Output files (same format as Phase 2 baseline outputs):
    data/processed/llama3_3b_finetuned_metrics.json   — aggregate scores
    data/processed/llama3_3b_finetuned_prompts.json   — per-row generated text + scores
    data/processed/candidate_baseline_comparison.csv  — appended row for comparison
"""

import json
import os
import warnings
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.evaluation.evaluate import (
    compute_aggregates,
    format_prompt,
    generate_response,
    run_row_loop,
    save_results,
)
from scripts.evaluation.metrics import bootstrap_ci
from models.base_utils.common_utils import read_params

warnings.filterwarnings("ignore")

BASE_MODEL_ID  = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_ID     = "taljindergill78/indian-recipe-llama3.2-qlora"

CONFIG = {
    "max_new_tokens": 768,     # increased from 512 — fine-tuned model generates longer recipes
    "do_sample": True,         # sampling breaks repetition loops that greedy decoding causes
    "load_in_4bit": torch.cuda.is_available(),
    "generation_kwargs": {
        "temperature": 0.7,        # controls randomness — 0.7 is focused but not deterministic
        "top_p": 0.9,              # nucleus sampling — ignores the bottom 10% of probability mass
        "repetition_penalty": 1.3, # penalises repeating the same tokens — directly prevents the
                                   # ingredient repetition loop seen with greedy decoding
    },
}

OUTPUT_METRICS = "data/processed/llama3_3b_finetuned_metrics.json"
OUTPUT_PROMPTS = "data/processed/llama3_3b_finetuned_prompts.json"
COMPARISON_CSV = "data/processed/candidate_baseline_comparison.csv"


def load_finetuned_model():
    print(f"Loading base model: {BASE_MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )

    print(f"Loading LoRA adapter: {ADAPTER_ID}")
    model = PeftModel.from_pretrained(base, ADAPTER_ID)
    model.eval()

    print(f"Loading tokenizer: {ADAPTER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main():
    params = read_params("params.yaml")
    test_path = params["data_dir"]["test"]

    print(f"\n{'='*60}")
    print(f"Evaluating fine-tuned model: {ADAPTER_ID}")
    print(f"Base model:   {BASE_MODEL_ID}")
    print(f"Test set:     {test_path}")
    print(f"Outputs →     {OUTPUT_PROMPTS}")
    print(f"Metrics →     {OUTPUT_METRICS}")
    print(f"{'='*60}\n")

    test_df = pd.read_csv(test_path)
    print(f"Loaded {len(test_df)} test rows.")

    model, tokenizer = load_finetuned_model()

    row_results = run_row_loop(model, tokenizer, test_df, CONFIG)

    # Checkpoint immediately in case aggregation fails
    checkpoint_path = OUTPUT_PROMPTS.replace(".json", "_rowcheckpoint.json")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(row_results, f)
    print(f"Row results checkpointed: {checkpoint_path}")

    aggregate = compute_aggregates(row_results, f"{BASE_MODEL_ID} + {ADAPTER_ID}")

    print("\n--- Fine-Tuned Model Results ---")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")

    save_results(row_results, aggregate, OUTPUT_PROMPTS, OUTPUT_METRICS, COMPARISON_CSV)
    print("\nFine-tuned evaluation complete.")


if __name__ == "__main__":
    main()
