"""
Run once after cloning to download all raw datasets from HuggingFace into data/raw/.

Usage:
    uv run python scripts/download_data.py
"""
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from models.base_utils.common_utils import read_params

config = read_params("params.yaml")


def download_anupam_dataset():
    """
    Anupam007/indian-recipe-dataset — 5,938 rows from archanaskitchen.com.
    Columns: TranslatedRecipeName, TranslatedIngredients, TotalTimeInMins,
             Cuisine, TranslatedInstructions, URL, Cleaned-Ingredients,
             image-url, Ingredient-count
    """
    output_path = config["data_dir"]["indian_data"]
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists.")
        return

    print("Downloading Anupam007/indian-recipe-dataset...")
    ds = load_dataset("Anupam007/indian-recipe-dataset", split="train")
    df = ds.to_pandas()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} rows → {output_path}")


def download_nf_analyst_dataset():
    """
    nf-analyst/indian_recipe_dataset — 6,871 rows from archanaskitchen.com.
    Extra columns vs Anupam007: Diet, Course, PrepTimeInMins, CookTimeInMins, Servings.
    Missing vs Anupam007: TranslatedRecipeName, Cleaned-Ingredients, URL, image-url.
    Raw text column is parsed into structured columns on download.
    """
    output_path = config["data_dir"]["nf_analyst_data"]
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists.")
        return

    print("Downloading nf-analyst/indian_recipe_dataset...")
    ds = load_dataset("nf-analyst/indian_recipe_dataset", split="train")
    df = ds.to_pandas()

    # Dataset stores all fields in a single 'text' column separated by ###
    def parse_text(text):
        parts = re.split(r"###\s*", text)
        record = {}
        for part in parts:
            if ":" in part:
                key, _, val = part.partition(":")
                record[key.strip()] = val.strip()
        return record

    import pandas as pd
    df = pd.DataFrame(df["text"].apply(parse_text).tolist())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} rows → {output_path}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    download_anupam_dataset()
    download_nf_analyst_dataset()
    print("\nAll datasets downloaded. Run scripts/eda.py to explore.")
