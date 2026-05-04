"""
Filter both raw recipe datasets to Indian-cuisine rows only.

Reads:  data/raw/indian_recipe_dataset.csv
        data/raw/nf_analyst_recipes_raw.csv
Writes: data/processed/anupam_filtered.csv
        data/processed/nf_analyst_filtered.csv

Usage:
    uv run python scripts/pipeline/filter_data.py
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.base_utils.common_utils import read_params

config = read_params("params.yaml")

# 42 cuisine values classified as Indian after full EDA review.
# Fusion excluded — style inconsistency risk for fine-tuning.
# Course labels (Appetizer, Snack, Dinner, etc.) excluded — data quality errors in source.
INDIAN_CUISINES = {
    "Indian", "North Indian Recipes", "South Indian Recipes", "Bengali Recipes",
    "Maharashtrian Recipes", "Kerala Recipes", "Tamil Nadu", "Karnataka", "Andhra",
    "Rajasthani", "Gujarati Recipes", "Goan Recipes", "Punjabi", "Chettinad",
    "Kashmiri", "Mangalorean", "Indo Chinese", "Parsi Recipes", "Awadhi", "Konkan",
    "Sindhi", "Oriya Recipes", "Mughlai", "Assamese", "Bihari", "Hyderabadi",
    "North East India Recipes", "Himachal", "Udupi", "Coorg", "North Karnataka",
    "Uttar Pradesh", "Coastal Karnataka", "Malabar", "Lucknowi", "South Karnataka",
    "Malvani", "Nagaland", "Uttarakhand-North Kumaon", "Kongunadu", "Haryana", "Jharkhand",
}


def filter_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    # Strip BOM and whitespace so "Gujarati Recipes﻿" matches "Gujarati Recipes"
    df["Cuisine"] = df["Cuisine"].str.strip().str.replace("﻿", "", regex=False)

    before = len(df)
    filtered = df[df["Cuisine"].isin(INDIAN_CUISINES)].copy()
    after = len(filtered)

    print(f"\n{name}")
    print(f"  Before : {before:,} rows")
    print(f"  After  : {after:,} rows  ({before - after:,} non-Indian dropped, "
          f"{after/before*100:.1f}% retained)")
    print(f"  Cuisine values kept: {filtered['Cuisine'].nunique()} | "
          f"dropped: {df[~df['Cuisine'].isin(INDIAN_CUISINES)]['Cuisine'].nunique()}")
    return filtered


def main():
    os.makedirs(config["data_dir"]["processed"], exist_ok=True)

    df_anupam = pd.read_csv(config["data_dir"]["indian_data"])
    df_anupam_filtered = filter_dataset(df_anupam, "Anupam007 (indian_recipe_dataset.csv)")
    df_anupam_filtered.to_csv(config["data_dir"]["anupam_filtered"], index=False)
    print(f"  Saved → {config['data_dir']['anupam_filtered']}")

    df_nf = pd.read_csv(config["data_dir"]["nf_analyst_data"])
    df_nf_filtered = filter_dataset(df_nf, "nf-analyst (nf_analyst_recipes_raw.csv)")
    df_nf_filtered.to_csv(config["data_dir"]["nf_analyst_filtered"], index=False)
    print(f"  Saved → {config['data_dir']['nf_analyst_filtered']}")

    print(f"\nTotal rows available for merge: "
          f"{len(df_anupam_filtered) + len(df_nf_filtered):,} (overlap resolved in next step)")
    print("Next: run scripts/pipeline/merge_and_clean.py")


if __name__ == "__main__":
    main()
