"""
Unit tests for scripts/evaluation/parser.py

Run with:  uv run pytest tests/test_parser.py -v
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluation.parser import parse, ParsedRecipe

# ──────────────────────────────────────────────
# Fixtures — sample model outputs
# ──────────────────────────────────────────────

PERFECT_FORMAT = """\
**Dal Makhani**

**Ingredients:**
1 cup whole black lentils, 2 tbsp butter, 1 cup cream, 2 tomatoes, 1 tsp cumin seeds

**Instructions:**
Soak the lentils overnight in water.
Drain and pressure cook for 4-5 whistles until soft.
In a pan, heat butter and add cumin seeds."""

MULTILINE_INGREDIENTS = """\
**Palak Paneer**

**Ingredients:**
- 200g paneer cubes
- 3 cups spinach leaves
- 2 tomatoes, chopped
- 1 tsp cumin seeds
- 1 tsp garam masala

**Instructions:**
Blanch spinach and blend to a smooth paste.
Fry paneer cubes until golden."""

FALLBACK_FORMAT = """\
### Palak Paneer

### Ingredients
200g paneer, 3 cups spinach, 2 tomatoes, 1 tsp cumin

### Instructions
Blanch spinach. Blend to paste. Fry paneer."""

LOWERCASE_HEADERS = """\
**Butter Chicken**

**ingredients:**
500g chicken, 2 tbsp butter, 1 cup tomato puree, 1 tsp garam masala

**instructions:**
Marinate chicken in yogurt. Cook in butter."""

MISSING_NAME = """\
**Ingredients:**
1 cup rice, 2 cups water, 1 tsp salt

**Instructions:**
Boil water. Add rice. Cook for 20 minutes."""

EMPTY_OUTPUT = ""

GARBAGE_OUTPUT = "This is not a recipe. Just some random text."


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def test_perfect_format_name():
    result = parse(PERFECT_FORMAT)
    assert result.name == "Dal Makhani"

def test_perfect_format_ingredients_count():
    result = parse(PERFECT_FORMAT)
    assert len(result.ingredients) == 5

def test_perfect_format_instructions_nonempty():
    result = parse(PERFECT_FORMAT)
    assert len(result.instructions) > 20

def test_multiline_ingredients():
    result = parse(MULTILINE_INGREDIENTS)
    assert result.name == "Palak Paneer"
    assert len(result.ingredients) >= 5

def test_fallback_format_ingredients():
    """Regex fallback must extract ingredients when ### headers are used."""
    result = parse(FALLBACK_FORMAT)
    assert len(result.ingredients) >= 3

def test_fallback_format_instructions():
    result = parse(FALLBACK_FORMAT)
    assert "Blanch" in result.instructions

def test_lowercase_headers():
    result = parse(LOWERCASE_HEADERS)
    assert len(result.ingredients) >= 3
    assert "Marinate" in result.instructions

def test_missing_name_returns_ingredients():
    """Even without a recipe name, ingredients and instructions should parse."""
    result = parse(MISSING_NAME)
    assert len(result.ingredients) >= 2
    assert "Boil" in result.instructions

def test_empty_output_returns_empty_recipe():
    result = parse(EMPTY_OUTPUT)
    assert result.is_empty()

def test_garbage_output_does_not_crash():
    """Parser must never raise an exception."""
    result = parse(GARBAGE_OUTPUT)
    assert isinstance(result, ParsedRecipe)

def test_ingredients_stripped_of_dashes():
    """Bullet markers should be removed from ingredient strings."""
    result = parse(MULTILINE_INGREDIENTS)
    for ingr in result.ingredients:
        assert not ingr.startswith("-")
        assert not ingr.startswith("*")
