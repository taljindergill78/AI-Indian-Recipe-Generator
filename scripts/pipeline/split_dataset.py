"""
Split prompt-response pairs into train, validation, and test sets.

Split sizes are read from params.yaml (split_sizes.test and split_sizes.val).
The test set is fixed so Phase 2 (baseline evaluation) and Phase 3 (post-fine-tune
evaluation) use the exact same 500 recipes — enabling a clean before/after comparison.

Reads:  data/processed/prompt_response_pairs.csv
Writes: data/processed/train.csv
        data/processed/val.csv
        data/processed/test.csv

Usage:
    uv run python scripts/pipeline/split_dataset.py
"""
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.base_utils.common_utils import read_params

config = read_params("params.yaml")

TEST_SIZE = config["split_sizes"]["test"]    # 500 — fixed, never changes
VAL_SIZE  = config["split_sizes"]["val"]     # 250


def main():
    df = pd.read_csv(config["data_dir"]["prompt_response_pairs"])
    total = len(df)
    print(f"Loaded {total:,} rows from prompt_response_pairs.csv")

    # Step 1: carve out the test set first (fixed at TEST_SIZE rows)
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=42,
        shuffle=True,
    )

    # Step 2: split the remainder into train and val
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE,
        random_state=42,
        shuffle=True,
    )

    print(f"\n  Train : {len(train):,} rows  ({len(train)/total*100:.1f}%)")
    print(f"  Val   : {len(val):,} rows  ({len(val)/total*100:.1f}%)")
    print(f"  Test  : {len(test):,} rows  ({len(test)/total*100:.1f}%)")
    print(f"  Total : {len(train) + len(val) + len(test):,} rows  ✅")

    # Verify no overlap between splits
    train_idx = set(train.index)
    val_idx   = set(val.index)
    test_idx  = set(test.index)
    assert len(train_idx & val_idx) == 0,  "Train/Val overlap detected!"
    assert len(train_idx & test_idx) == 0, "Train/Test overlap detected!"
    assert len(val_idx & test_idx) == 0,   "Val/Test overlap detected!"
    print("\n  No overlap between splits ✅")

    # Diet distribution check — ensure test set has good coverage
    print("\n  Diet distribution in test set:")
    diet_counts = test["diet"].value_counts(dropna=False)
    for diet, count in diet_counts.items():
        print(f"    {count:4d}  {diet}")

    train.to_csv(config["data_dir"]["train"], index=False)
    val.to_csv(config["data_dir"]["val"],     index=False)
    test.to_csv(config["data_dir"]["test"],   index=False)

    print(f"\n✅ Saved:")
    print(f"   {config['data_dir']['train']}")
    print(f"   {config['data_dir']['val']}")
    print(f"   {config['data_dir']['test']}")
    print("\nPhase 1 complete. Phase 2: run baseline evaluation on all 3 model candidates.")


if __name__ == "__main__":
    main()
