"""
Recompute evaluation metrics from stored generated_text using the fixed parser.

WHY THIS EXISTS:
  The original Kaggle evaluation run used an older parser.py that had two bugs:
    1. Bold sub-headers inside instructions (e.g. **Step 1: Marinate**) reset
       current_section to None, causing gen_instructions to be empty for rows
       where the model used numbered step headers.
    2. The Tier 2 regex couldn't consume the closing ** in **Instructions:**,
       causing fallback to also fail on those rows.

  This deflated ROUGE-L, BLEU, and BERTScore in candidate_baseline_comparison.csv.
  The generated_text (raw model output) is stored correctly in *_base_prompts.json.
  This script re-parses that text with the fixed parser and recomputes all metrics.

WHAT IT PRODUCES:
  - Updated *_base_prompts.json files (re-parsed fields + corrected per-row scores)
  - Updated *_base_metrics.json files (corrected aggregate metrics)
  - Updated candidate_baseline_comparison.csv (corrected comparison table)

RUN FROM PROJECT ROOT:
  uv run python scripts/evaluation/recompute_metrics.py

NOTE ON BERTSCORE:
  BERTScore downloads ~500MB of RoBERTa weights on first run (cached after that).
  On CPU (Mac) this takes ~10-20 minutes for 500 rows. On MPS (Apple Silicon) it
  is faster. The script auto-detects the best available device.
  Skip BERTScore recompute with --skip-bertscore if you only need ROUGE-L / BLEU / F1.

CLI USAGE:
  uv run python scripts/evaluation/recompute_metrics.py
  uv run python scripts/evaluation/recompute_metrics.py --skip-bertscore
  uv run python scripts/evaluation/recompute_metrics.py --model llama3_3b
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.base_utils.common_utils import read_params
from scripts.evaluation.parser import parse
from scripts.evaluation.metrics import (
    bootstrap_ci,
    compute_bertscore,
    compute_bleu_corpus,
    compute_bleu_sentence,
    compute_ingredient_f1,
    compute_rouge_l,
)

params = read_params("params.yaml")

MODEL_KEYS = ["llama3_3b", "llama3_1b", "phi3_mini"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-row recompute
# ─────────────────────────────────────────────────────────────────────────────

def _recompute_row(row: dict) -> dict:
    """Re-parse generated_text and reference_text; recompute per-row metrics."""
    gen_parsed = parse(row["generated_text"])
    ref_parsed = parse(row["reference_text"])

    f1_result = compute_ingredient_f1(gen_parsed.ingredients, ref_parsed.ingredients)
    rouge_l = compute_rouge_l(gen_parsed.instructions, ref_parsed.instructions)
    bleu_s = compute_bleu_sentence(gen_parsed.instructions, ref_parsed.instructions)

    return {
        **row,
        "gen_name": gen_parsed.name,
        "gen_ingredients": gen_parsed.ingredients,
        "gen_instructions": gen_parsed.instructions,
        "ref_name": ref_parsed.name,
        "ref_ingredients": ref_parsed.ingredients,
        "ref_instructions": ref_parsed.instructions,
        "ingredient_f1": f1_result["f1"],
        "ingredient_precision": f1_result["precision"],
        "ingredient_recall": f1_result["recall"],
        "rouge_l": rouge_l,
        "bleu_sentence": bleu_s,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate recompute
# ─────────────────────────────────────────────────────────────────────────────

def _compute_aggregate(
    rows: list[dict],
    model_id: str,
    skip_bertscore: bool,
    bertscore_device: str,
) -> dict:
    ingredient_f1s = [r["ingredient_f1"] for r in rows]
    rouge_ls = [r["rouge_l"] for r in rows]
    gen_instr = [r["gen_instructions"] for r in rows]
    ref_instr = [r["ref_instructions"] for r in rows]

    bleu_corpus = compute_bleu_corpus(gen_instr, ref_instr)

    ingr_ci = bootstrap_ci(ingredient_f1s)
    rouge_ci = bootstrap_ci(rouge_ls)

    aggregate = {
        "model_id": model_id,
        "n_samples": len(rows),
        "ingredient_f1_mean": round(float(np.mean(ingredient_f1s)), 4),
        "ingredient_f1_ci_lower": ingr_ci[0],
        "ingredient_f1_ci_upper": ingr_ci[1],
        "rouge_l_mean": round(float(np.mean(rouge_ls)), 4),
        "rouge_l_ci_lower": rouge_ci[0],
        "rouge_l_ci_upper": rouge_ci[1],
        "bleu_corpus": bleu_corpus,
    }

    if skip_bertscore:
        aggregate.update({
            "bertscore_mean_f1": None,
            "bertscore_mean_precision": None,
            "bertscore_mean_recall": None,
            "bertscore_ci_lower": None,
            "bertscore_ci_upper": None,
        })
    else:
        print(f"  Computing BERTScore on {bertscore_device} (may take 10-20 min on CPU)...")
        bs = compute_bertscore(gen_instr, ref_instr, device=bertscore_device)
        bert_ci = bootstrap_ci(bs["per_sample_f1"])
        aggregate.update({
            "bertscore_mean_f1": bs["mean_f1"],
            "bertscore_mean_precision": bs["mean_precision"],
            "bertscore_mean_recall": bs["mean_recall"],
            "bertscore_ci_lower": bert_ci[0],
            "bertscore_ci_upper": bert_ci[1],
        })

    return aggregate


# ─────────────────────────────────────────────────────────────────────────────
# CSV update (same logic as evaluate.py save_results)
# ─────────────────────────────────────────────────────────────────────────────

def _update_comparison_csv(aggregate: dict) -> None:
    csv_path = Path("data/processed/candidate_baseline_comparison.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = {k: v for k, v in aggregate.items() if v is not None}

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["model_id"] != aggregate["model_id"]]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(csv_path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main(model_keys: list[str], skip_bertscore: bool) -> None:
    device = _detect_device()
    print(f"Device: {device}")

    for model_key in model_keys:
        model_id = params["model_candidates"][model_key]
        prompts_path = Path(params["data_path"][f"{model_key}_prompts"])
        metrics_path = Path(params["data_path"][f"{model_key}_metrics"])

        print(f"\n{'='*60}")
        print(f"Model: {model_key} ({model_id})")
        print(f"Loading: {prompts_path}")

        if not prompts_path.exists():
            print(f"  SKIP — {prompts_path} not found")
            continue

        with open(prompts_path, encoding="utf-8") as f:
            rows = json.load(f)

        print(f"  Loaded {len(rows)} rows. Re-parsing with fixed parser...")

        # Count parse failures before recompute
        empty_before = sum(1 for r in rows if not r.get("gen_instructions", ""))
        print(f"  Empty gen_instructions BEFORE recompute: {empty_before}/{len(rows)} ({100*empty_before/len(rows):.1f}%)")

        rows = [_recompute_row(r) for r in rows]

        empty_after = sum(1 for r in rows if not r.get("gen_instructions", ""))
        print(f"  Empty gen_instructions AFTER  recompute: {empty_after}/{len(rows)} ({100*empty_after/len(rows):.1f}%)")

        old_rouge = sum(r.get("rouge_l", 0) for r in rows) / len(rows)  # after recompute
        print(f"  New ROUGE-L mean: {old_rouge:.4f}")

        aggregate = _compute_aggregate(rows, model_id, skip_bertscore, device)

        print(f"\n  Aggregate results:")
        print(f"    Ingredient F1 : {aggregate['ingredient_f1_mean']:.4f}  CI [{aggregate['ingredient_f1_ci_lower']:.4f}, {aggregate['ingredient_f1_ci_upper']:.4f}]")
        print(f"    ROUGE-L       : {aggregate['rouge_l_mean']:.4f}  CI [{aggregate['rouge_l_ci_lower']:.4f}, {aggregate['rouge_l_ci_upper']:.4f}]")
        print(f"    Corpus BLEU   : {aggregate['bleu_corpus']:.4f}")
        if not skip_bertscore:
            print(f"    BERTScore F1  : {aggregate['bertscore_mean_f1']:.4f}  CI [{aggregate['bertscore_ci_lower']:.4f}, {aggregate['bertscore_ci_upper']:.4f}]")

        print(f"\n  Saving updated prompts → {prompts_path}")
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(rows, f)

        print(f"  Saving updated metrics → {metrics_path}")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)

        _update_comparison_csv(aggregate)
        print(f"  Updated candidate_baseline_comparison.csv")

    print("\n✅ Done. Run 'uv run python scripts/evaluation/recompute_metrics.py' again to verify.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(
        description="Recompute evaluation metrics from stored generated_text using the fixed parser.",
    )
    cli.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore recompute (fast — only ROUGE-L, BLEU, Ingredient F1).",
    )
    cli.add_argument(
        "--model",
        choices=MODEL_KEYS,
        default=None,
        help="Recompute only one model (default: all three).",
    )
    args = cli.parse_args()

    keys = [args.model] if args.model else MODEL_KEYS
    main(keys, args.skip_bertscore)
