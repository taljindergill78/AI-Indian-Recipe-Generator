"""
Run once to download the raw dataset from HuggingFace into data/raw/.

Usage:
    uv run python scripts/download_data.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from models.base_utils.common_utils import read_params

config = read_params("params.yaml")
output_path = config["data_dir"]["indian_data"]

if os.path.exists(output_path):
    print(f"Dataset already exists at {output_path} — skipping download.")
    sys.exit(0)

print("Downloading dataset from HuggingFace (Anupam007/indian-recipe-dataset)...")
ds = load_dataset("Anupam007/indian-recipe-dataset", split="train")
df = ds.to_pandas()

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
print(f"Columns: {list(df.columns)}")
