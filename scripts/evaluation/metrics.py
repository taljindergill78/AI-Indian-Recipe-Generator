"""
Metric computation for recipe generation evaluation.

Four metrics are computed:
  1. Ingredient F1   — set-overlap precision/recall/F1 on extracted ingredient lists
  2. ROUGE-L         — longest-common-subsequence F1 on instruction text
  3. BLEU-4          — n-gram precision on instruction text
                        sentence_bleu (per-sample) + corpus_bleu (aggregate)
  4. BERTScore       — semantic similarity on instruction text (batched over all samples)

All per-sample functions accept strings/lists and return plain Python floats or dicts.
bootstrap_ci() computes 95% confidence intervals from a list of per-sample scores.

Usage:
    from scripts.evaluation.metrics import (
        compute_ingredient_f1,
        compute_rouge_l,
        compute_bleu_sentence,
        compute_bleu_corpus,
        compute_bertscore,
        bootstrap_ci,
    )
"""

import re
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rs


# ─────────────────────────────────────────────────────────────────────────────
# Helper: ingredient normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_ingredient(s: str) -> str:
    """
    Lowercase, strip leading quantities/numbers, strip punctuation edges.
    "1 cup whole black lentils" → "whole black lentils"
    "2 tbsp butter"             → "butter"
    """
    s = s.lower().strip()
    # Remove leading quantity patterns: "1 cup", "2 tbsp", "3-4", "½ tsp", etc.
    s = re.sub(r"^[\d½¼¾\s/.-]+(cup|tbsp|tsp|g|kg|ml|l|oz|lb|clove|cloves|inch|piece|pieces|pinch|handful|bunch|sprig|sprigs|medium|large|small)s?\s+", "", s)
    # Remove any remaining leading digits and fractions
    s = re.sub(r"^[\d½¼¾\s/.-]+", "", s).strip()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ingredient F1
# ─────────────────────────────────────────────────────────────────────────────

def compute_ingredient_f1(
    generated_ingredients: list[str],
    reference_ingredients: list[str],
) -> dict:
    """
    Compute precision, recall, and F1 on the ingredient lists.

    Normalises both lists (lowercase, strip quantities) before comparison.
    Uses set intersection — order does not matter.

    Returns:
        {"precision": float, "recall": float, "f1": float}
        All values are 0.0 if either list is empty.
    """
    if not generated_ingredients or not reference_ingredients:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    gen_set = {_normalise_ingredient(i) for i in generated_ingredients if i.strip()}
    ref_set = {_normalise_ingredient(i) for i in reference_ingredients if i.strip()}

    # Remove empty strings that survive normalisation
    gen_set.discard("")
    ref_set.discard("")

    if not gen_set or not ref_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = len(gen_set & ref_set)
    precision = overlap / len(gen_set)
    recall    = overlap / len(ref_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROUGE-L
# ─────────────────────────────────────────────────────────────────────────────

_rouge_scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rouge_l(generated: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between generated and reference instruction text.

    use_stemmer=True: reduces words to their stems before comparison
    ("cooking" == "cook" == "cooked"), giving fairer credit for recipe verbs.

    Returns float in [0, 1].  Returns 0.0 if either string is empty.
    """
    if not generated or not generated.strip():
        return 0.0
    if not reference or not reference.strip():
        return 0.0

    score = _rouge_scorer.score(reference, generated)
    return round(score["rougeL"].fmeasure, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 3. BLEU-4
# ─────────────────────────────────────────────────────────────────────────────

_smoothing = SmoothingFunction().method1


def compute_bleu_sentence(generated: str, reference: str) -> float:
    """
    Per-sample BLEU-4 using sentence_bleu with smoothing.

    Smoothing (method1): adds a small count to 0-count n-grams so short texts
    don't get a BLEU of 0 just because their 4-grams didn't match.

    Use this for per-sample JSON output only.
    For the aggregate metric, use compute_bleu_corpus() — it is statistically sounder.

    Returns float in [0, 1].
    """
    if not generated or not reference:
        return 0.0

    ref_tokens = reference.split()
    hyp_tokens = generated.split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_smoothing)
    return round(float(score), 4)


def compute_bleu_corpus(
    generated_list: list[str],
    reference_list: list[str],
) -> float:
    """
    Aggregate BLEU-4 over all samples using corpus_bleu.

    corpus_bleu accumulates n-gram counts across ALL examples before dividing —
    this is the statistically correct approach for evaluating a model on a test set.
    It avoids the instability of averaging per-sample sentence_bleu scores.

    Returns float in [0, 1].
    """
    if not generated_list or not reference_list:
        return 0.0

    list_of_references = [[ref.split()] for ref in reference_list]
    hypotheses         = [gen.split() for gen in generated_list]

    score = corpus_bleu(list_of_references, hypotheses)
    return round(float(score), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore(
    generated_list: list[str],
    reference_list: list[str],
    lang: str = "en",
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """
    Compute BERTScore F1 over all samples in a single batched call.

    BERTScore measures semantic similarity using contextual BERT embeddings.
    Running batched over all 500 samples is ~15x faster than one-at-a-time.

    First run downloads ~500MB (the bert-base-uncased weights) and caches them.
    Subsequent runs use the cache.

    Args:
        generated_list: list of generated instruction strings (one per test row)
        reference_list: list of reference instruction strings (one per test row)
        lang: language code — "en" selects bert-base-uncased automatically
        batch_size: how many pairs to process per GPU/CPU forward pass
        device: "cuda" on Kaggle/GPU, "cpu" for local testing

    Returns:
        {
            "per_sample_f1": [0.87, 0.91, 0.83, ...],   # one score per row
            "mean_f1":  0.87,
            "mean_precision": 0.86,
            "mean_recall": 0.88,
        }
    """
    # Import here so the heavy bert_score library only loads when this function is called
    from bert_score import score as bert_score_fn

    if not generated_list or not reference_list:
        return {"per_sample_f1": [], "mean_f1": 0.0, "mean_precision": 0.0, "mean_recall": 0.0}

    P, R, F1 = bert_score_fn(
        generated_list,
        reference_list,
        lang=lang,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )

    per_f1 = [round(f, 4) for f in F1.tolist()]
    return {
        "per_sample_f1":   per_f1,
        "mean_f1":         round(float(F1.mean()), 4),
        "mean_precision":  round(float(P.mean()),  4),
        "mean_recall":     round(float(R.mean()),  4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Compute a bootstrap confidence interval for the mean of a list of scores.

    Algorithm:
      1. Resample len(scores) values from the list WITH replacement → 1 bootstrap sample
      2. Compute the mean of that sample
      3. Repeat n_bootstrap times
      4. Sort the bootstrap means
      5. Return the (1-ci)/2 and (1+ci)/2 percentiles as (lower, upper)

    A 95% CI means: if you repeated your evaluation on a different random set of
    500 recipes, the true mean metric would fall in this interval 95% of the time.

    Returns (lower_bound, upper_bound), both rounded to 4 decimal places.
    Returns (0.0, 0.0) if scores list is empty.
    """
    if not scores:
        return (0.0, 0.0)

    arr = np.array(scores, dtype=float)
    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility

    bootstrap_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    lower_pct = (1.0 - ci) / 2.0 * 100
    upper_pct = (1.0 + ci) / 2.0 * 100

    lower = float(np.percentile(bootstrap_means, lower_pct))
    upper = float(np.percentile(bootstrap_means, upper_pct))

    return round(lower, 4), round(upper, 4)
