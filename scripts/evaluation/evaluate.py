"""
Phase 2 Model Selection — Evaluation Orchestrator

Runs a HuggingFace model through all 500 test rows, collects per-row metrics,
computes aggregate scores with bootstrap CIs, saves results, and logs to MLflow.

Usage (change model_key in CONFIG to switch candidates):
    uv run python scripts/evaluation/evaluate.py

On Kaggle: upload this file, set CONFIG["model_key"] and run.
"""

import json
import os
import warnings
from pathlib import Path

import mlflow
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.evaluation.metrics import (
    bootstrap_ci,
    compute_bertscore,
    compute_bleu_corpus,
    compute_bleu_sentence,
    compute_ingredient_f1,
    compute_rouge_l,
)
from scripts.evaluation.parser import parse
from models.base_utils.common_utils import read_params

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit model_key to switch candidates
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Key must match a key under model_candidates in params.yaml
    # Options: "llama3_3b" | "llama3_1b" | "phi3_mini"
    "model_key": "llama3_3b",

    "max_new_tokens": 512,
    "do_sample": False,          # greedy decoding — reproducible across runs

    # 4-bit quantization (True on Kaggle T4/P100, False for CPU local testing)
    "load_in_4bit": torch.cuda.is_available(),

    "mlflow_experiment": "phase2_model_selection",
    "mlflow_tracking_uri": "mlruns",    # local directory; override on Kaggle if needed
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str, load_in_4bit: bool):
    """
    Load model and tokenizer from HuggingFace hub.

    4-bit quantization: loads weights as 4-bit integers, dequantises to bfloat16
    during the forward pass. ~4x memory reduction vs full precision.
    Required to fit a 3B model on a Kaggle T4 (16GB VRAM).

    load_in_4bit=False for local CPU testing — loads in full float32 (slow but works).
    """
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print(f"Loading model in 4-bit: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        print(f"Loading model in float32 (CPU): {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_prompt(row: pd.Series, tokenizer) -> str:
    """
    Convert a test row into the model's input format using apply_chat_template.

    apply_chat_template adds model-specific special tokens (e.g., <|begin_of_text|>,
    [INST], <|user|>) that the model was pre-trained on. Without these tokens,
    the model doesn't know where the user message starts and the response begins.

    add_generation_prompt=True appends the assistant turn opener so the model
    immediately starts generating the recipe (not another user message).
    """
    messages = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user",   "content": row["user_message"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
) -> str:
    """
    Run one forward pass through the model and decode the generated tokens.

    @torch.inference_mode(): disables gradient tracking entirely during generation.
    This is ~20% faster and uses less VRAM than torch.no_grad(), because it also
    disables autograd's version counter. Always use this for evaluation loops.

    We decode only the NEW tokens (output_ids[:, input_len:]) to strip the input
    prompt from the decoded text. model.generate() returns the full sequence
    (input + output) by default.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Per-row evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_row_loop(model, tokenizer, test_df: pd.DataFrame, config: dict) -> list[dict]:
    """
    Iterate over all test rows, generate a response for each, parse both
    generated and reference, and compute per-row metrics.

    Returns a list of row result dicts — one per test row.
    Failures (OOM, generation error) store empty strings and zero metrics
    so the loop never stops mid-way through 500 rows.
    """
    row_results = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        result = {
            "row_index":          int(idx),
            "user_message":       row["user_message"],
            "reference_text":     row["assistant_response"],
            "generated_text":     "",
            "ref_name":           "",
            "ref_ingredients":    [],
            "ref_instructions":   "",
            "gen_name":           "",
            "gen_ingredients":    [],
            "gen_instructions":   "",
            "ingredient_f1":      0.0,
            "ingredient_precision": 0.0,
            "ingredient_recall":  0.0,
            "rouge_l":            0.0,
            "bleu_sentence":      0.0,
        }

        try:
            prompt = format_prompt(row, tokenizer)
            generated_text = generate_response(
                model, tokenizer, prompt,
                config["max_new_tokens"],
                config["do_sample"],
            )
            result["generated_text"] = generated_text

            ref_parsed = parse(row["assistant_response"])
            gen_parsed = parse(generated_text)

            result.update({
                "ref_name":         ref_parsed.name,
                "ref_ingredients":  ref_parsed.ingredients,
                "ref_instructions": ref_parsed.instructions,
                "gen_name":         gen_parsed.name,
                "gen_ingredients":  gen_parsed.ingredients,
                "gen_instructions": gen_parsed.instructions,
            })

            f1_scores = compute_ingredient_f1(gen_parsed.ingredients, ref_parsed.ingredients)
            result.update({
                "ingredient_f1":        f1_scores["f1"],
                "ingredient_precision": f1_scores["precision"],
                "ingredient_recall":    f1_scores["recall"],
                "rouge_l":              compute_rouge_l(gen_parsed.instructions, ref_parsed.instructions),
                "bleu_sentence":        compute_bleu_sentence(gen_parsed.instructions, ref_parsed.instructions),
            })

        except Exception as e:
            print(f"\n[WARNING] Row {idx} failed: {e}. Storing zeros and continuing.")

        row_results.append(result)

    return row_results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_aggregates(row_results: list[dict], model_id: str) -> dict:
    """
    Compute corpus-level metrics and bootstrap confidence intervals from row results.

    Three metrics need the full list of samples rather than averaging per-row scores:
    - corpus_bleu: accumulates n-gram counts across ALL rows before dividing
    - BERTScore: runs in a single batched forward pass (15x faster than per-row)
    - bootstrap CI: needs all per-sample scores to resample from

    Everything else is the mean of per-row values.
    """
    gen_instructions = [r["gen_instructions"] for r in row_results]
    ref_instructions = [r["ref_instructions"] for r in row_results]
    ing_f1_scores    = [r["ingredient_f1"]  for r in row_results]
    rouge_l_scores   = [r["rouge_l"]        for r in row_results]

    print("Computing corpus BLEU...")
    bleu_corpus = compute_bleu_corpus(gen_instructions, ref_instructions)

    print("Computing BERTScore (this takes a few minutes on CPU)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_results = compute_bertscore(gen_instructions, ref_instructions, device=device)

    print("Computing bootstrap confidence intervals...")
    ing_f1_ci   = bootstrap_ci(ing_f1_scores)
    rouge_l_ci  = bootstrap_ci(rouge_l_scores)
    bertscore_ci = bootstrap_ci(bert_results["per_sample_f1"])

    n = len(row_results)
    return {
        "model_id":                model_id,
        "n_samples":               n,

        "ingredient_f1_mean":      round(sum(ing_f1_scores) / n, 4),
        "ingredient_f1_ci_lower":  ing_f1_ci[0],
        "ingredient_f1_ci_upper":  ing_f1_ci[1],

        "rouge_l_mean":            round(sum(rouge_l_scores) / n, 4),
        "rouge_l_ci_lower":        rouge_l_ci[0],
        "rouge_l_ci_upper":        rouge_l_ci[1],

        "bleu_corpus":             bleu_corpus,

        "bertscore_mean_f1":       bert_results["mean_f1"],
        "bertscore_mean_precision": bert_results["mean_precision"],
        "bertscore_mean_recall":   bert_results["mean_recall"],
        "bertscore_ci_lower":      bertscore_ci[0],
        "bertscore_ci_upper":      bertscore_ci[1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Saving results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    row_results: list[dict],
    aggregate: dict,
    outputs_path: str,
    metrics_path: str,
    comparison_csv_path: str,
) -> None:
    """
    Save three output files:

    1. outputs_path (*_prompts.json): per-row details — generated text, parsed fields,
       per-row scores. Use this to inspect which recipes scored lowest.

    2. metrics_path (*_metrics.json): aggregate scores for this candidate.

    3. comparison_csv_path (candidate_baseline_comparison.csv): one row per candidate.
       Append if the file exists (so running candidate 2 doesn't erase candidate 1's row).
    """
    Path(outputs_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving per-row outputs → {outputs_path}")
    with open(outputs_path, "w") as f:
        json.dump(row_results, f, indent=2)

    print(f"Saving aggregate metrics → {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Append one row to the comparison CSV
    agg_row = pd.DataFrame([aggregate])
    if Path(comparison_csv_path).exists():
        existing = pd.read_csv(comparison_csv_path)
        # Replace row if this model_id already has an entry (re-run scenario)
        existing = existing[existing["model_id"] != aggregate["model_id"]]
        updated = pd.concat([existing, agg_row], ignore_index=True)
    else:
        updated = agg_row

    updated.to_csv(comparison_csv_path, index=False)
    print(f"Updated comparison table → {comparison_csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MLflow logging
# ─────────────────────────────────────────────────────────────────────────────

def log_to_mlflow(aggregate: dict, config: dict, params: dict) -> None:
    """
    Log one MLflow run per candidate evaluation.

    Parameters logged: model_id, max_new_tokens, test_size, quantization
    Metrics logged: all aggregate scores and CI bounds

    MLflow run name = model_id so you can identify runs in the UI.
    """
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])

    with mlflow.start_run(run_name=aggregate["model_id"]):
        mlflow.log_params({
            "model_id":       aggregate["model_id"],
            "max_new_tokens": config["max_new_tokens"],
            "test_size":      aggregate["n_samples"],
            "load_in_4bit":   config["load_in_4bit"],
            "do_sample":      config["do_sample"],
        })

        metrics_to_log = {k: v for k, v in aggregate.items()
                          if isinstance(v, float) or isinstance(v, int)}
        metrics_to_log.pop("n_samples", None)
        mlflow.log_metrics(metrics_to_log)

    print(f"MLflow run logged under experiment '{config['mlflow_experiment']}'")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    params = read_params("params.yaml")

    model_key = CONFIG["model_key"]
    model_id  = params["model_candidates"][model_key]

    test_path       = params["data_dir"]["test"]
    # params.yaml uses "_prompts" for per-row output JSON (stores generated + reference text)
    outputs_path = params["data_path"][f"{model_key}_prompts"]
    metrics_path = params["data_path"][f"{model_key}_metrics"]
    comparison_path = "data/processed/candidate_baseline_comparison.csv"

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_id}")
    print(f"Test set:   {test_path}")
    print(f"Outputs →   {outputs_path}")
    print(f"Metrics →   {metrics_path}")
    print(f"{'='*60}\n")

    test_df = pd.read_csv(test_path)
    print(f"Loaded {len(test_df)} test rows.")

    model, tokenizer = load_model_and_tokenizer(model_id, CONFIG["load_in_4bit"])

    row_results = run_row_loop(model, tokenizer, test_df, CONFIG)

    aggregate = compute_aggregates(row_results, model_id)

    print("\n--- Aggregate Results ---")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")

    save_results(row_results, aggregate, outputs_path, metrics_path, comparison_path)
    log_to_mlflow(aggregate, CONFIG, params)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
