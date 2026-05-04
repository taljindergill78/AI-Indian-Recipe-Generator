"""
Merge, enrich, and clean the two filtered recipe datasets into one unified dataset.

Strategy:
  1. LEFT JOIN Anupam007 <- nf-analyst on first-80-char instruction prefix
     Fills Diet, Course, PrepTimeInMins, CookTimeInMins, Servings into Anupam007 rows.
  2. CONCAT nf-analyst-only rows (recipes not found in Anupam007).
  3. DROP 6 redundant columns (see docs/decisions/merged_dataset_column_selection.md).
  4. DROP rows without recipe names for training consistency
     (see docs/decisions/drop_nameless_rows.md).
  5. CLEAN: null ingredients, duplicates, zero TotalTimeInMins, whitespace.

Reads:  data/processed/anupam_filtered.csv
        data/processed/nf_analyst_filtered.csv
Writes: data/processed/merged_cleaned.csv

Usage:
    uv run python scripts/pipeline/merge_and_clean.py
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.base_utils.common_utils import read_params

config = read_params("params.yaml")
SEP = "=" * 60

# Columns kept after column audit — see docs/decisions/merged_dataset_column_selection.md
KEEP_COLS = [
    "TranslatedRecipeName",
    "TranslatedIngredients",
    "TotalTimeInMins",
    "Servings",
    "Cuisine",
    "Course",
    "Diet",
    "TranslatedInstructions",
]


def make_join_key(series: pd.Series) -> pd.Series:
    """First 80 stripped chars of instruction text — used as a dedup / join key."""
    return series.str.strip().str[:80]


def merge_datasets(df_anupam: pd.DataFrame, df_nf: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}")
    print("STEP 1 — MERGE (left join + concat nf-only rows)")
    print(SEP)

    df_anupam = df_anupam.copy()
    df_nf = df_nf.copy()

    df_anupam["_key"] = make_join_key(df_anupam["TranslatedInstructions"])
    df_nf["_key"] = make_join_key(df_nf["TranslatedInstructions"])

    enrichment_cols = ["_key", "Diet", "Course", "PrepTimeInMins", "CookTimeInMins", "Servings"]
    nf_enrichment = df_nf[enrichment_cols].drop_duplicates(subset="_key")

    df_merged = df_anupam.merge(nf_enrichment, on="_key", how="left")
    matched = df_merged["Diet"].notna().sum()
    print(f"  Anupam007 rows:            {len(df_anupam):,}")
    print(f"  Enriched with Diet/Course: {matched:,}")
    print(f"  Anupam007-only (no match): {len(df_anupam) - matched:,}")

    nf_only_mask = ~df_nf["_key"].isin(df_anupam["_key"].values)
    df_nf_only = df_nf[nf_only_mask].copy()
    print(f"\n  nf-analyst-only recipes:   {len(df_nf_only):,}")

    for col in ["TranslatedRecipeName", "Cleaned-Ingredients", "URL", "image-url", "Ingredient-count"]:
        df_nf_only[col] = None

    all_cols = [
        "TranslatedRecipeName", "TranslatedIngredients", "Cleaned-Ingredients",
        "TotalTimeInMins", "PrepTimeInMins", "CookTimeInMins", "Servings",
        "Cuisine", "Course", "Diet", "TranslatedInstructions",
        "URL", "image-url", "Ingredient-count",
    ]
    df_merged = df_merged.reindex(columns=all_cols)
    df_nf_only = df_nf_only.reindex(columns=all_cols)

    combined = pd.concat([df_merged, df_nf_only], ignore_index=True)
    print(f"\n  Combined rows (pre-clean):  {len(combined):,}")
    return combined


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}")
    print("STEP 2 — DROP REDUNDANT COLUMNS")
    print(SEP)

    cols_to_drop = [c for c in ["Cleaned-Ingredients", "PrepTimeInMins",
                                 "CookTimeInMins", "URL", "image-url", "Ingredient-count"]
                    if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped: {cols_to_drop}")
    print(f"  Remaining columns: {list(df.columns)}")

    print(f"\n{SEP}")
    print("STEP 3 — DROP ROWS WITHOUT RECIPE NAMES")
    print(SEP)

    before = len(df)
    df = df[df["TranslatedRecipeName"].notna() & (df["TranslatedRecipeName"].str.strip() != "")]
    print(f"  Dropped {before - len(df):,} nameless rows → {len(df):,} rows remain")

    print(f"\n{SEP}")
    print("STEP 4 — CLEAN")
    print(SEP)

    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates dropped:           {before - len(df)}")

    before = len(df)
    df = df[df["TranslatedIngredients"].notna() & (df["TranslatedIngredients"].str.strip() != "")]
    print(f"  Null-ingredient rows dropped: {before - len(df)}")

    before = len(df)
    df = df[df["TranslatedInstructions"].notna() & (df["TranslatedInstructions"].str.strip() != "")]
    print(f"  Null-instruction rows dropped:{before - len(df)}")

    df["TotalTimeInMins"] = pd.to_numeric(df["TotalTimeInMins"], errors="coerce")
    still_zero = df["TotalTimeInMins"] == 0
    df.loc[still_zero, "TotalTimeInMins"] = None
    print(f"  Zero TotalTimeInMins → NaN:   {still_zero.sum()}")

    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    df = df.reset_index(drop=True)
    print(f"\n  Final row count: {len(df):,}")
    return df


def profile_merged(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("FINAL DATASET PROFILE")
    print(SEP)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n  Null counts:")
    for col, n in df.isnull().sum().items():
        flag = "⚠️ " if n > 0 else "✅"
        print(f"    {flag} {col}: {n:,}  ({n/len(df)*100:.1f}%)")

    print(f"\n  Has Diet label:  {df['Diet'].notna().sum():,} rows ({df['Diet'].notna().mean()*100:.1f}%)")
    print(f"  Has Course label:{df['Course'].notna().sum():,} rows ({df['Course'].notna().mean()*100:.1f}%)")

    print("\n  Diet distribution:")
    for diet, count in df["Diet"].value_counts().items():
        print(f"    {count:5d}  {diet}")


def main():
    os.makedirs(config["data_dir"]["processed"], exist_ok=True)

    df_anupam = pd.read_csv(config["data_dir"]["anupam_filtered"])
    df_nf = pd.read_csv(config["data_dir"]["nf_analyst_filtered"])

    combined = merge_datasets(df_anupam, df_nf)
    cleaned = clean_dataset(combined)
    profile_merged(cleaned)

    cleaned.to_csv(config["data_dir"]["merged_cleaned"], index=False)
    print(f"\n✅ Saved → {config['data_dir']['merged_cleaned']}")
    print("Next: run scripts/pipeline/format_prompts.py")


if __name__ == "__main__":
    main()
