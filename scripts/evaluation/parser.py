"""
Parse a model's recipe output into structured fields: name, ingredients, instructions.

The model is trained to produce output in this format:
    **RecipeName**

    **Ingredients:**
    1 cup black lentils, 2 tbsp butter, ...

    **Instructions:**
    Soak the lentils overnight. Pressure cook...

Parsing is two-tier:
  Tier 1 — structured line-by-line parsing (fast, matches the training format exactly)
  Tier 2 — regex fallback (permissive, handles minor format deviations from the model)

This module is used in two contexts:
  1. Parsing generated outputs during evaluation (compare generated vs reference)
  2. Parsing reference outputs from test.csv assistant_response column (the ground truth)

CLI (project root):  uv run python scripts/evaluation/parser.py
                      uv run python scripts/evaluation/parser.py -f path/to/recipe.txt
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedRecipe:
    """
    Structured output of the parser.
    All fields default to empty so callers can check is_empty() rather than catch exceptions.
    """
    name: str = ""
    ingredients: list = field(default_factory=list)
    instructions: str = ""

    def is_empty(self) -> bool:
        return not self.name and not self.ingredients and not self.instructions


def _split_ingredients(raw: str) -> list[str]:
    """
    Split a raw ingredient block into individual ingredient strings.
    Handles both comma-separated (inline) and newline-separated (listed) formats.
    Strips leading list markers like '-' or '*'.
    """
    # Try splitting on newlines first (listed format)
    lines = [l.strip() for l in raw.split("\n") if l.strip()]

    if len(lines) > 1:
        # Multi-line format: each line is one ingredient (possibly prefixed with '-' or '*')
        items = []
        for line in lines:
            item = re.sub(r"^[-*•]\s*", "", line).strip()
            if item:
                # Further split by comma in case a line has multiple ingredients
                for part in item.split(","):
                    part = part.strip()
                    if part:
                        items.append(part)
        return items
    else:
        # Single-line comma-separated format
        return [i.strip() for i in raw.split(",") if i.strip()]


def _parse_structured(text: str) -> ParsedRecipe:
    """
    Tier 1: parse using the exact **Header:** markers from training.
    Returns a ParsedRecipe (some fields may be empty if not found).
    """
    lines = text.strip().split("\n")
    name = ""

    # Recipe name: first non-empty bold line  **Name**
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
            candidate = stripped.strip("*").strip()
            # Exclude section headers like "**Ingredients:**"
            if "ingredient" not in candidate.lower() and "instruction" not in candidate.lower():
                name = candidate
                break

    # Walk lines to find Ingredients and Instructions sections
    current_section = None
    ingr_lines: list[str] = []
    instr_lines: list[str] = []

    for line in lines:
        lower = line.lower().strip()
        is_bold_header = line.strip().startswith("**")

        if is_bold_header and "ingredient" in lower:
            current_section = "ingredients"
            continue
        elif is_bold_header and "instruction" in lower:
            current_section = "instructions"
            continue
        elif is_bold_header and line.strip().endswith("**"):
            # Any other bold header resets the section
            current_section = None
            continue

        if current_section == "ingredients" and line.strip():
            ingr_lines.append(line.strip())
        elif current_section == "instructions" and line.strip():
            instr_lines.append(line.strip())

    ingredients = _split_ingredients("\n".join(ingr_lines)) if ingr_lines else []
    instructions = "\n".join(instr_lines)

    return ParsedRecipe(name=name, ingredients=ingredients, instructions=instructions)


def _parse_regex_fallback(text: str) -> ParsedRecipe:
    """
    Tier 2: regex-based fallback for models that use non-bold headers
    (e.g. '### Ingredients', 'Ingredients:', '## Instructions').
    """
    # Match an ingredient section header (bold, hash, or plain) followed by its content
    ingr_pattern = re.compile(
        r"(?:#{1,3}\s*|[*_]{1,2})?ingredients?(?:[*_]{1,2})?:?\s*\n(.*?)"
        r"(?=\n\s*(?:#{1,3}\s*|[*_]{1,2})?instructions?|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    instr_pattern = re.compile(
        r"(?:#{1,3}\s*|[*_]{1,2})?instructions?(?:[*_]{1,2})?:?\s*\n(.*?)$",
        re.IGNORECASE | re.DOTALL,
    )

    ingr_match = ingr_pattern.search(text)
    instr_match = instr_pattern.search(text)

    ingredients = _split_ingredients(ingr_match.group(1).strip()) if ingr_match else []
    instructions = instr_match.group(1).strip() if instr_match else ""

    return ParsedRecipe(name="", ingredients=ingredients, instructions=instructions)


def parse(text: str) -> ParsedRecipe:
    """
    Parse a model's recipe output into a ParsedRecipe.

    Strategy:
      1. Run structured Tier-1 parser (exact **Header:** matching).
      2. For any field Tier 1 missed, try the regex Tier-2 fallback.
      3. Return the best-of-both result.

    Never raises an exception — returns an empty ParsedRecipe on complete failure.
    """
    if not text or not text.strip():
        return ParsedRecipe()

    result = _parse_structured(text)

    # Patch missing fields with fallback
    if not result.ingredients or not result.instructions:
        fallback = _parse_regex_fallback(text)
        if not result.ingredients:
            result.ingredients = fallback.ingredients
        if not result.instructions:
            result.instructions = fallback.instructions

    return result


def _demo_sample() -> str:
    """Short recipe-shaped string for exploring parse() from the CLI without a file."""
    return """\
**Dal Makhani**

**Ingredients:**
1 cup whole black lentils, 2 tbsp butter, 1 cup cream

**Instructions:**
Soak the lentils overnight in water.
Drain and pressure cook for 4-5 whistles until soft."""


def _print_parsed(result: ParsedRecipe) -> None:
    print(result)
    if result.name:
        print("name:", result.name)
    if result.ingredients:
        print("ingredients:")
        for item in result.ingredients:
            print(f"  - {item}")
    if result.instructions:
        print("instructions:")
        print(result.instructions)


def main() -> None:
    cli = argparse.ArgumentParser(
        description="Run parse() on recipe-shaped text (demo, or text from a file).",
    )
    cli.add_argument(
        "-f",
        "--file",
        type=Path,
        metavar="PATH",
        help="UTF-8 file containing model-style recipe output. If omitted, uses a built-in demo.",
    )
    args = cli.parse_args()
    raw = args.file.read_text(encoding="utf-8") if args.file else _demo_sample()
    _print_parsed(parse(raw))


if __name__ == "__main__":
    main()
