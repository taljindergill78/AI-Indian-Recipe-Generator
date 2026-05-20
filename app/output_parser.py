"""
Parses raw model output into a structured dict that the recipe card renders.

The fine-tuned model was trained on this exact format (from format_prompts.py):
    **RecipeName**

    **Ingredients:**
    ingredient text (comma-separated or line-by-line)

    **Instructions:**
    step-by-step text (numbered or paragraph)

Parser strategy:
  1. Try structured regex extraction (fast, reliable for well-formed output)
  2. Fall back to fuzzy section matching (handles minor format deviations)
  3. Return whatever was extractable; UI renders partial output gracefully

The parsed dict always has the same keys — missing fields get sensible defaults.
"""
import re


def parse(raw_text: str) -> dict:
    """
    Convert raw model output string to a structured recipe dict.

    Returns a dict with keys:
        name (str), ingredients (list[str]), steps (list[str]),
        metadata (dict): {region, diet, time_mins, spice_level, course}
    """
    text = raw_text.strip()

    name = _extract_name(text)
    ingredients = _extract_ingredients(text)
    steps = _extract_steps(text)
    metadata = _extract_metadata(text)

    return {
        "name": name,
        "ingredients": ingredients,
        "steps": steps,
        "metadata": metadata,
        "raw": raw_text,
    }


def _extract_name(text: str) -> str:
    # Primary: bold heading at the start — **Name**
    match = re.search(r"^\*\*(.+?)\*\*", text, re.MULTILINE)
    if match:
        candidate = match.group(1).strip()
        # Reject if it's a section header like "Ingredients:" or "Instructions:"
        if not any(kw in candidate.lower() for kw in ["ingredient", "instruction", "step", "method"]):
            return candidate

    # Fallback: first non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return lines[0].lstrip("*#").strip()

    return "Indian Recipe"


def _extract_ingredients(text: str) -> list[str]:
    # Find the Ingredients section
    ing_pattern = re.search(
        r"\*\*Ingredients[:\s]*\*\*(.*?)(?=\*\*Instructions|\*\*Method|\*\*Steps|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not ing_pattern:
        # Try without bold markers
        ing_pattern = re.search(
            r"Ingredients[:\s]+(.*?)(?=Instructions|Method|Steps|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )

    if not ing_pattern:
        return []

    ing_block = ing_pattern.group(1).strip()

    # The training data uses comma-separated ingredients on one or multiple lines
    # OR newline-separated with bullet dashes
    items: list[str] = []

    if "\n" in ing_block and any(line.strip().startswith(("-", "•", "*")) for line in ing_block.split("\n")):
        # Bullet list format
        for line in ing_block.split("\n"):
            item = line.strip().lstrip("-•*").strip()
            if item and len(item) > 1:
                items.append(item)
    else:
        # Comma-separated (most common in training data).
        # Don't split on commas inside parentheses — e.g. "500g Chicken (boneless, cut into pieces)"
        # stays as one ingredient, not two.
        raw_items = re.split(r",\s*(?![^(]*\))", ing_block.replace("\n", " "))
        items = [i.strip() for i in raw_items if i.strip() and len(i.strip()) > 1]

    # Deduplicate while preserving order
    seen: set[str] = set()
    clean: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen and len(item) < 120:
            seen.add(key)
            clean.append(item)

    return clean[:30]  # Cap at 30 ingredients


def _extract_steps(text: str) -> list[str]:
    # Find the Instructions section
    inst_pattern = re.search(
        r"\*\*(?:Instructions|Method|Steps)[:\s]*\*\*(.*?)(?=\*\*(?:Tips|Serving|Notes|Chef)|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not inst_pattern:
        inst_pattern = re.search(
            r"(?:Instructions|Method|Steps)[:\s]+(.*?)(?=Tips|Serving|Notes|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )

    if not inst_pattern:
        # Last resort: everything after the ingredients section
        parts = re.split(r"\*\*Ingredients[:\s]*\*\*", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            remainder = parts[1]
        else:
            remainder = text

        sentences = re.split(r"(?<=[.!?])\s+", remainder.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20][:15]

    inst_block = inst_pattern.group(1).strip()

    steps: list[str] = []

    # Try numbered steps first: "1. ...", "Step 1:", "1) ..."
    numbered = re.findall(r"(?:^|\n)\s*(?:Step\s*)?\d+[.):\s]+(.+?)(?=(?:\n\s*(?:Step\s*)?\d+[.):\s])|\Z)", inst_block, re.DOTALL)
    if numbered:
        for step in numbered:
            clean = step.strip().replace("\n", " ")
            if len(clean) > 15:
                steps.append(clean)
        return steps[:15]

    # Sentence splitting fallback
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", inst_block.replace("\n", " "))
    return [s.strip() for s in sentences if len(s.strip()) > 20][:15]


def _extract_metadata(text: str) -> dict:
    """Extract optional metadata fields the model may or may not include."""
    meta = {
        "region": None,
        "diet": None,
        "time_mins": None,
        "spice_level": None,
        "course": None,
    }

    # Region patterns: "North Indian", "Bengali", "South Indian", etc.
    region_match = re.search(
        r"\b(North Indian|South Indian|Bengali|Gujarati|Maharashtrian|Rajasthani|"
        r"Punjabi|Goan|Kerala|Tamil Nadu|Andhra|Karnataka|Mughlai|Kashmiri|"
        r"Hyderabadi|Awadhi|Chettinad|Konkan|Sindhi|Parsi)\b",
        text, re.IGNORECASE
    )
    if region_match:
        meta["region"] = region_match.group(1)

    # Diet
    diet_match = re.search(
        r"\b(Vegetarian|Vegan|Non[- ]Vegetarian|Non[- ]Veg|Jain|Diabetic)\b",
        text, re.IGNORECASE
    )
    if diet_match:
        raw = diet_match.group(1).lower()
        if "vegan" in raw:
            meta["diet"] = "Vegan"
        elif "jain" in raw:
            meta["diet"] = "Jain"
        elif "non" in raw:
            meta["diet"] = "Non-Vegetarian"
        else:
            meta["diet"] = "Vegetarian"

    # Time: look for total recipe time labels only — NOT step-level durations like "cook 2 minutes"
    # Patterns: "Time: 35 mins", "Total Time: 1 hour", "Ready in 30 minutes", "Prep: 45 mins"
    time_match = re.search(
        r"(?:total\s+time|prep\s+time|cooking\s+time|time[:\s]|ready\s+in)[:\s]+(\d+)\s*(?:to\s*\d+\s*)?(?:minutes?|mins?|hours?|hrs?)",
        text, re.IGNORECASE
    )
    if time_match:
        val = int(time_match.group(1))
        if "hour" in time_match.group(0).lower():
            val *= 60
        meta["time_mins"] = val

    # Spice level from chilli count (🌶️×N or "spice level: N")
    spice_match = re.search(r"[Ss]pice\s+[Ll]evel[:\s]+(\d)", text)
    if spice_match:
        meta["spice_level"] = min(5, int(spice_match.group(1)))

    # Course
    course_match = re.search(
        r"\b(Main Course|Side Dish|Dessert|Snack|Breakfast|Starter|Soup|Salad|Bread|Drink)\b",
        text, re.IGNORECASE
    )
    if course_match:
        meta["course"] = course_match.group(1)

    return meta
