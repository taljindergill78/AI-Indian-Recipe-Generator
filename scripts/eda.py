"""
Exploratory Data Analysis on all raw datasets.

Usage:
    uv run python scripts/eda.py
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_utils.common_utils import read_params

config = read_params("params.yaml")

SEP = "=" * 60


def profile_dataset(df, name):
    print(f"\n{SEP}")
    print(f"DATASET: {name}")
    print(SEP)

    print(f"\n[1] Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"    Columns: {list(df.columns)}")

    print("\n[2] Null values per column:")
    for col, count in df.isnull().sum().items():
        status = "⚠️ " if count > 0 else "✅"
        print(f"    {status} {col}: {count}")

    print(f"\n[3] Duplicate rows: {df.duplicated().sum()}")

    if "Cuisine" in df.columns:
        print(f"\n[4] Cuisine — {df['Cuisine'].nunique()} unique values (top 20):")
        for cuisine, count in df["Cuisine"].value_counts().head(20).items():
            tag = " ← Indian" if any(
                kw in cuisine.lower() for kw in [
                    "indian", "south indian", "north indian", "andhra", "bengali",
                    "gujarati", "punjabi", "rajasthani", "kerala", "tamil",
                    "maharashtrian", "mughal", "awadhi", "hyderabadi",
                    "karnataka", "goan", "kashmiri", "chettinad", "biryani"
                ]
            ) else ""
            print(f"    {count:5d}  {cuisine}{tag}")

    if "TranslatedInstructions" in df.columns:
        lengths = df["TranslatedInstructions"].dropna().str.split().str.len()
        print(f"\n[5] Instruction length (words): "
              f"min={lengths.min()}, median={lengths.median():.0f}, "
              f"mean={lengths.mean():.0f}, max={lengths.max()}, p95={lengths.quantile(0.95):.0f}")

    if "TotalTimeInMins" in df.columns:
        t = df["TotalTimeInMins"]
        print(f"\n[6] TotalTimeInMins: min={t.min()}, median={t.median():.0f}, "
              f"mean={t.mean():.0f}, max={t.max()}, zeros={( t== 0).sum()}")

    if "Diet" in df.columns:
        print(f"\n[7] Diet distribution:")
        for diet, count in df["Diet"].value_counts().items():
            print(f"    {count:5d}  {diet}")

    if "Course" in df.columns:
        print(f"\n[8] Course distribution:")
        for course, count in df["Course"].value_counts().items():
            print(f"    {count:5d}  {course}")

    if "Ingredient-count" in df.columns:
        ic = df["Ingredient-count"]
        print(f"\n[9] Ingredient count: min={ic.min()}, median={ic.median():.0f}, "
              f"mean={ic.mean():.0f}, max={ic.max()}")


def overlap_analysis(df_a, df_b, name_a, name_b):
    print(f"\n{SEP}")
    print(f"OVERLAP: {name_a}  vs  {name_b}")
    print(SEP)
    instr_a = set(df_a["TranslatedInstructions"].str[:80].str.strip())
    instr_b = set(df_b["TranslatedInstructions"].str[:80].str.strip())
    overlap = instr_a & instr_b
    only_a = instr_a - instr_b
    only_b = instr_b - instr_a
    print(f"  {name_a}: {len(instr_a)} unique recipes")
    print(f"  {name_b}: {len(instr_b)} unique recipes")
    print(f"  Overlap:        {len(overlap)} recipes in both")
    print(f"  Only in {name_a}: {len(only_a)}")
    print(f"  Only in {name_b}: {len(only_b)}")
    print(f"  Combined unique: {len(instr_a | instr_b)}")


# --- Load datasets ---
df_anupam = pd.read_csv(config["data_dir"]["indian_data"])
df_nf = pd.read_csv(config["data_dir"]["nf_analyst_data"])

# --- Profile each dataset ---
profile_dataset(df_anupam, "Anupam007 (indian_recipe_dataset.csv)")
profile_dataset(df_nf, "nf-analyst (nf_analyst_recipes_raw.csv)")

# --- Overlap ---
overlap_analysis(df_anupam, df_nf, "Anupam007", "nf-analyst")

print(f"\n{SEP}")
print("EDA complete.")
print(SEP)
