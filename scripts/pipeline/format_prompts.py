"""
Convert the merged recipe dataset into model-agnostic prompt-response pairs for SFT.

Each row produces two columns:
  user_message      — the user-turn content (cuisine, diet, course, recipe name)
  assistant_response — the assistant-turn content (name header, ingredients, instructions)

No model-specific tokens are added here. The training script applies
tokenizer.apply_chat_template() for whichever model is being fine-tuned.
The system prompt is read from params.yaml so it can be changed without editing code.

Reads:  data/processed/merged_cleaned.csv
Writes: data/processed/prompt_response_pairs.csv

Usage:
    uv run python scripts/pipeline/format_prompts.py
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.base_utils.common_utils import read_params

config = read_params("params.yaml")
SYSTEM_PROMPT = config["prompts"]["system_prompt"].strip()


def build_user_message(row: pd.Series) -> str:
    """
    Constructs the user-turn content from available fields.
    Only non-null optional fields (Diet, Course) are included.

    Examples:
      "Generate a Vegetarian North Indian dinner recipe for Dal Makhani"
      "Generate a North Indian Recipes recipe for Butter Naan"  (no Diet/Course)
    """
    descriptor_parts = []

    if pd.notna(row.get("Diet")) and str(row["Diet"]).strip():
        descriptor_parts.append(str(row["Diet"]).strip())

    descriptor_parts.append(str(row["Cuisine"]).strip())

    descriptor = " ".join(descriptor_parts)
    article = "an" if descriptor[0].lower() in "aeiou" else "a"
    parts = [f"Generate {article}", descriptor]

    if pd.notna(row.get("Course")) and str(row["Course"]).strip():
        parts.append(str(row["Course"]).lower().strip())

    parts.append(f"recipe for {str(row['TranslatedRecipeName']).strip()}")

    return " ".join(parts)


def build_assistant_response(row: pd.Series) -> str:
    """
    Constructs the assistant-turn content.
    Format:
      **RecipeName**

      **Ingredients:**
      ...

      **Instructions:**
      ...
    """
    sections = [f"**{str(row['TranslatedRecipeName']).strip()}**\n"]
    sections.append(f"**Ingredients:**\n{str(row['TranslatedIngredients']).strip()}")
    sections.append(f"**Instructions:**\n{str(row['TranslatedInstructions']).strip()}")
    return "\n\n".join(sections)


def main():
    df = pd.read_csv(config["data_dir"]["merged_cleaned"])
    print(f"Loaded {len(df):,} rows from merged_cleaned.csv")
    print(f"System prompt: {SYSTEM_PROMPT[:80]}...")

    records = []
    for _, row in df.iterrows():
        records.append({
            "system_prompt": SYSTEM_PROMPT,
            "user_message": build_user_message(row),
            "assistant_response": build_assistant_response(row),
            "cuisine": row["Cuisine"],
            "diet": row.get("Diet"),
            "course": row.get("Course"),
        })

    df_out = pd.DataFrame(records)

    empty_user = (df_out["user_message"].str.strip() == "").sum()
    empty_asst = (df_out["assistant_response"].str.strip() == "").sum()
    print(f"  Empty user messages:       {empty_user}")
    print(f"  Empty assistant responses: {empty_asst}")
    print(f"  Rows with Diet label:      {df_out['diet'].notna().sum():,}")

    print("\nSample user message (row 0):")
    print(" ", df_out.iloc[0]["user_message"])
    print("\nSample assistant response (row 0, first 200 chars):")
    print(" ", df_out.iloc[0]["assistant_response"][:200])

    out_path = os.path.join(config["data_dir"]["processed"], "prompt_response_pairs.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\n✅ Saved {len(df_out):,} prompt-response pairs → {out_path}")
    print("Next: run scripts/pipeline/split_dataset.py")


if __name__ == "__main__":
    main()
