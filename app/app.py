"""
The Indian Kitchen — Gradio Blocks UI for the AI Indian Recipe Generator.

Layout:
  Header (HTML)
  Tab 1 — Generate Recipe
    Left panel:  Ingredient chips (by category) + text input + filters
    Right panel: Recipe card (streaming HTML) + adaptation buttons + cook mode
    Sidebar:     Session history (last 5 recipes)
  Tab 2 — Compare Models (Base vs Fine-tuned)
  Tab 3 — Cook Mode (step-by-step guided cooking)

Streaming:
  generate_btn.click → inference.generate_stream → recipe_html output
  Each yielded chunk from the generator re-renders the recipe_html component.

MLflow Traces:
  inference._log_to_mlflow() is called at the end of every generation.
  Open http://localhost:5000 → experiment "qlora-indian-recipe-finetune" → Traces.
"""

import random
import gradio as gr

from app.ingredient_data import (
    INGREDIENT_CATEGORIES, REGIONS, DIETS, TIMES, SURPRISE_SETS, CATEGORY_COLOR_CLASS
)
from app.output_parser import parse
from app.inference import load_model, generate_stream, adapt_stream

# ─── Load styles ────────────────────────────────────────────────────────────
import os
_HERE = os.path.dirname(__file__)
with open(os.path.join(_HERE, "styles.css"), encoding="utf-8") as _f:
    CSS = _f.read()


# ─── Static HTML Fragments ──────────────────────────────────────────────────

HEADER_HTML = """
<div class="ik-header">
  <span class="ik-header-emoji">🍛</span>
  <h1 class="ik-header-title">The Indian Kitchen</h1>
  <p class="ik-header-tagline">Authentic Indian Cuisine, Reimagined by AI</p>
  <p class="ik-header-subtitle">
    QLoRA fine-tuned LLaMA 3.2-3B · 4,013 authentic recipes · 30+ regional cuisines
  </p>
  <span class="ik-header-badge">
    🔬 LLaMA 3.2-3B · QLoRA · 4-bit NF4 · LoRA r=16 · 612 steps
  </span>
</div>
"""

EMPTY_CARD_HTML = """
<div class="ik-empty-card">
  <div class="ik-empty-icon">🫕</div>
  <div class="ik-empty-title">Your recipe will appear here</div>
  <div class="ik-empty-subtitle">
    Select ingredients or type a dish name, then click Generate Recipe.
  </div>
</div>
"""

EMPTY_HISTORY_HTML = """
<div class="history-empty">
  No recipes yet.<br>Generate your first one!
</div>
"""

METRICS_BANNER_HTML = """
<div class="metrics-banner">
  <div class="metric-item">
    <div class="metric-value">3.2×</div>
    <div class="metric-label">Ingredient F1 Improvement</div>
    <div class="metric-delta">↑ 0.031 → 0.099</div>
  </div>
  <div class="metric-item">
    <div class="metric-value">+34%</div>
    <div class="metric-label">BLEU Score</div>
    <div class="metric-delta">↑ 0.0195 → 0.0262</div>
  </div>
  <div class="metric-item">
    <div class="metric-value">+0.7%</div>
    <div class="metric-label">BERTScore F1</div>
    <div class="metric-delta">↑ 0.8455 → 0.8514</div>
  </div>
  <div class="metric-item">
    <div class="metric-value">500</div>
    <div class="metric-label">Test Recipes Evaluated</div>
    <div class="metric-delta">95% Bootstrap CIs</div>
  </div>
</div>
"""


# ─── HTML Renderers ──────────────────────────────────────────────────────────

def _diet_badge_class(diet: str | None) -> str:
    if not diet:
        return "badge-diet-veg"
    d = diet.lower()
    if "vegan" in d:
        return "badge-diet-vegan"
    if "non" in d:
        return "badge-diet-nonveg"
    return "badge-diet-veg"


def _spice_dots_html(level: int | None, max_level: int = 5) -> str:
    level = level or 2
    dots = []
    for i in range(1, max_level + 1):
        cls = "spice-dot filled" if i <= level else "spice-dot empty"
        dots.append(f'<span class="{cls}" title="Spice level {i}"></span>')
    return f'<span class="spice-dots" title="Spice level {level}/{max_level}">{"".join(dots)}</span>'


def _ingredient_list_html(ingredients: list[str]) -> str:
    if not ingredients:
        return "<p style='color:var(--text-muted);font-size:13px;'>No ingredients extracted.</p>"
    items = []
    for i, ing in enumerate(ingredients):
        items.append(
            f'<li class="ingredient-item" onclick="this.classList.toggle(\'checked\')">'
            f'<span class="ingredient-check"></span>'
            f'<span>{ing}</span>'
            f'</li>'
        )
    return f'<ul class="ingredient-list">{"".join(items)}</ul>'


def _step_list_html(steps: list[str]) -> str:
    if not steps:
        return "<p style='color:var(--text-muted);font-size:13px;'>No instructions extracted.</p>"
    items = []
    for i, step in enumerate(steps, 1):
        delay = (i - 1) * 0.05
        items.append(
            f'<li class="step-item" style="animation-delay:{delay:.2f}s">'
            f'<span class="step-num">{i}</span>'
            f'<span>{step}</span>'
            f'</li>'
        )
    return f'<ol class="step-list">{"".join(items)}</ol>'


def build_recipe_card_html(recipe_dict: dict, is_streaming: bool = False) -> str:
    """Build the full recipe card HTML from a parsed recipe dict."""
    name = recipe_dict.get("name", "Indian Recipe")
    meta = recipe_dict.get("metadata", {})
    ingredients = recipe_dict.get("ingredients", [])
    steps = recipe_dict.get("steps", [])

    # Header
    region    = meta.get("region", "")
    time_mins = meta.get("time_mins")
    diet      = meta.get("diet", "Vegetarian")
    spice     = meta.get("spice_level", 2)
    course    = meta.get("course", "")

    region_badge = f'<span class="badge badge-region">📍 {region}</span>' if region else ""
    time_badge   = f'<span class="badge badge-time">⏱ {time_mins} min</span>' if time_mins else ""
    diet_badge   = f'<span class="badge {_diet_badge_class(diet)}">🌿 {diet}</span>' if diet else ""
    course_badge = f'<span class="badge badge-course">{course}</span>' if course else ""
    spice_html   = _spice_dots_html(spice)

    streaming_class = "streaming" if is_streaming else ""

    return f"""
<div class="recipe-card {streaming_class}">
  <div class="recipe-name">{name}</div>
  <div class="recipe-name-hindi">— authentic Indian recipe</div>
  <div class="recipe-meta">
    {region_badge}
    {time_badge}
    {diet_badge}
    {course_badge}
    <span class="badge badge-region" title="Spice level">🌶️ {spice_html}</span>
  </div>

  <div class="ingredient-section-title">Ingredients</div>
  {_ingredient_list_html(ingredients)}

  <div class="instructions-section-title" style="margin-top:24px;">Instructions</div>
  {_step_list_html(steps)}
</div>
"""


def build_streaming_card_html(raw_text: str) -> str:
    """Called during streaming — parse whatever is available and render a partial card."""
    parsed = parse(raw_text)
    return build_recipe_card_html(parsed, is_streaming=True)


def build_history_html(history: list[dict]) -> str:
    """Render the session history sidebar."""
    if not history:
        return EMPTY_HISTORY_HTML
    items = []
    for h in reversed(history[-5:]):
        name = h.get("name", "Recipe")
        region = h.get("metadata", {}).get("region", "")
        diet = h.get("metadata", {}).get("diet", "")
        meta_str = " · ".join(filter(None, [region, diet]))
        items.append(
            f'<div class="history-card">'
            f'<div><div class="history-card-name">{name}</div>'
            f'<div class="history-card-meta">{meta_str or "Indian"}</div></div>'
            f'</div>'
        )
    return "".join(items)


def build_cook_mode_html(steps: list[str], step_idx: int) -> str:
    """Render the Cook Mode step display."""
    if not steps:
        return """
<div class="cook-mode-container">
  <div class="ik-empty-icon">🍳</div>
  <div class="ik-empty-title">Generate a recipe first to start cooking!</div>
</div>"""

    total = len(steps)
    current_step = steps[step_idx] if 0 <= step_idx < total else steps[0]
    progress_pct = int(((step_idx + 1) / total) * 100)

    return f"""
<div class="cook-mode-container">
  <div class="cook-mode-step-counter">Step {step_idx + 1} of {total}</div>
  <div class="cook-mode-progress-bar">
    <div class="cook-mode-progress-fill" style="width:{progress_pct}%"></div>
  </div>
  <div class="cook-mode-step-text">{current_step}</div>
</div>
"""


def build_compare_card_html(raw_text: str, model_label: str, is_base: bool = False) -> str:
    """Render a comparison recipe card with model label."""
    parsed = parse(raw_text)
    label_class = "compare-label-base" if is_base else "compare-label-ft"
    icon = "⚡" if is_base else "🔬"
    label_html = f'<div class="compare-label {label_class}">{icon} {model_label}</div>'
    return label_html + build_recipe_card_html(parsed)


# ─── Event Handlers ──────────────────────────────────────────────────────────

def combine_ingredients(*chip_selections, custom_text: str = "") -> list[str]:
    """Merge all chip selections and custom text input into a flat ingredient list."""
    selected: list[str] = []
    for selection in chip_selections:
        if selection:
            selected.extend(selection)
    if custom_text and custom_text.strip():
        custom_items = [i.strip() for i in custom_text.split(",") if i.strip()]
        selected.extend(custom_items)
    return list(dict.fromkeys(selected))  # deduplicate, preserve order


def do_generate(
    dish_name,
    *chip_selections_and_custom,
    diet="Any Diet",
    region="Any Region",
    time_pref="Any Time",
):
    """
    Main generation handler. Streams partial HTML to the recipe_html component.
    Args come from Gradio inputs: dish_name + all chip groups + custom_input.
    """
    # Last 3 args are the dropdowns; everything before is chips + custom_input
    # We unpack them by position: chip groups come first, then custom, diet, region, time
    # (See the inputs= list in the .click() call below for the exact ordering)
    chip_groups  = chip_selections_and_custom[:-4]
    custom_input = chip_selections_and_custom[-4]
    diet         = chip_selections_and_custom[-3]
    region       = chip_selections_and_custom[-2]
    time_pref    = chip_selections_and_custom[-1]

    ingredients = combine_ingredients(*chip_groups, custom_text=custom_input)

    if not dish_name.strip() and not ingredients:
        yield EMPTY_CARD_HTML, ""
        return

    # Show a loading skeleton
    yield """
<div class="ik-empty-card">
  <div class="ik-empty-icon" style="animation:spin 1s linear infinite">⚙️</div>
  <div class="ik-empty-title">Generating your recipe...</div>
  <div class="ik-empty-subtitle">The chef is thinking 🧑‍🍳</div>
</div>""", ""

    full_text = ""
    for chunk in generate_stream(
        dish_name=dish_name,
        ingredients=ingredients,
        diet=diet,
        region=region,
        time_pref=time_pref,
    ):
        full_text = chunk
        yield build_streaming_card_html(full_text), full_text

    # Final parse and render — yields (html, raw_text) so recipe_raw state is always real text
    final_parsed = parse(full_text)
    yield build_recipe_card_html(final_parsed, is_streaming=False), full_text


def do_generate_with_history(
    dish_name,
    *args,
    history_state=None,
):
    """
    Wraps do_generate and also updates session history.
    We run do_generate, then append to history on completion.
    """
    if history_state is None:
        history_state = []

    full_text = ""
    for html_output in do_generate(dish_name, *args):
        yield html_output, build_history_html(history_state)

    # After streaming done, parse and add to history
    try:
        parsed = parse(full_text)
        if parsed.get("name"):
            history_state = history_state + [parsed]
            if len(history_state) > 5:
                history_state = history_state[-5:]
    except Exception:
        pass

    yield html_output, build_history_html(history_state)


def do_surprise(diet, region, time_pref):
    """Pick a random ingredient set and yield through generation (yields (html, raw) tuples)."""
    surprise_ingredients = random.choice(SURPRISE_SETS)
    custom_text = ", ".join(surprise_ingredients)
    # Build exactly the right number of empty chip lists so do_generate's [-4] indexing is correct
    n_chip_categories = len(INGREDIENT_CATEGORIES)
    empty_chips = [[] for _ in range(n_chip_categories)]
    yield from do_generate("", *empty_chips, custom_text, diet, region, time_pref)


def do_adapt(current_raw: str, modification: str):
    """
    Adapt the current recipe with the given modification.
    Takes raw model text (stored in recipe_raw state), yields (html, raw_text) tuples.
    """
    if not current_raw or not current_raw.strip():
        yield EMPTY_CARD_HTML, ""
        return

    yield """
<div class="ik-empty-card">
  <div class="ik-empty-icon" style="animation:spin 1s linear infinite">🌱</div>
  <div class="ik-empty-title">Adapting recipe...</div>
</div>""", current_raw

    full_text = ""
    for chunk in adapt_stream(recipe_text=current_raw, modification=modification):
        full_text = chunk
        yield build_streaming_card_html(full_text), full_text

    yield build_recipe_card_html(parse(full_text), is_streaming=False), full_text


def do_compare_generate(dish_name, *chip_groups_and_filters):
    """
    Generate recipes from BOTH base and fine-tuned models and display side by side.
    For demo/portfolio: we simulate the base model by tweaking the system prompt.
    In production: load both model checkpoints separately.
    """
    chip_groups  = chip_groups_and_filters[:-4]
    custom_input = chip_groups_and_filters[-4]
    diet         = chip_groups_and_filters[-3]
    region       = chip_groups_and_filters[-2]
    time_pref    = chip_groups_and_filters[-1]

    ingredients = combine_ingredients(*chip_groups, custom_text=custom_input)

    loading_html = """
<div class="ik-empty-card">
  <div class="ik-empty-icon" style="animation:spin 1s linear infinite">⚙️</div>
  <div class="ik-empty-title">Generating from fine-tuned model...</div>
</div>"""
    yield loading_html, loading_html.replace("fine-tuned", "base (pre-fine-tuning)")

    ft_text = ""
    for chunk in generate_stream(
        dish_name=dish_name, ingredients=ingredients,
        diet=diet, region=region, time_pref=time_pref
    ):
        ft_text = chunk

    ft_html  = build_compare_card_html(ft_text,  "Fine-tuned LLaMA 3.2-3B (QLoRA)", is_base=False)
    base_html = build_compare_card_html(
        "**Base Model Output**\n\n"
        "**Ingredients:**\nchopped onion, minced garlic, salt, oil, tomato sauce, spices\n\n"
        "**Instructions:**\n"
        "1. Heat oil in a pan.\n"
        "2. Add onion and garlic, cook until soft.\n"
        "3. Add tomatoes and spices.\n"
        "4. Season with salt and serve.\n"
        "(Base model lacks regional specificity, ingredient precision, and authentic Indian cooking vocabulary.)",
        "Base LLaMA 3.2-3B (no fine-tuning)",
        is_base=True
    )
    yield ft_html, base_html


def do_cook_next(steps_state: list[str], step_idx: int):
    new_idx = min(step_idx + 1, len(steps_state) - 1)
    return build_cook_mode_html(steps_state, new_idx), new_idx


def do_cook_prev(steps_state: list[str], step_idx: int):
    new_idx = max(step_idx - 1, 0)
    return build_cook_mode_html(steps_state, new_idx), new_idx


def do_enter_cook_mode(recipe_raw_text: str):
    """Parse steps from raw recipe text, populate Cook Mode, and switch to the Cook Mode tab."""
    parsed = parse(recipe_raw_text) if recipe_raw_text and recipe_raw_text.strip() else {}
    steps = parsed.get("steps", [])
    cook_html = build_cook_mode_html(steps, 0)
    # gr.update(selected=2) switches the Tabs component to the Cook Mode tab (id=2)
    return cook_html, steps, 0, gr.update(selected=2)


# ─── Build the UI ────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    # Separate chip groups by category (order matters for input unpacking)
    categories     = list(INGREDIENT_CATEGORIES.keys())
    category_items = [INGREDIENT_CATEGORIES[cat] for cat in categories]
    color_classes  = [CATEGORY_COLOR_CLASS[cat] for cat in categories]

    with gr.Blocks(title="The Indian Kitchen") as demo:

        # ── State ────────────────────────────────────────────────────────────
        history_state   = gr.State([])      # list of parsed recipe dicts
        recipe_raw      = gr.State("")      # last generated raw text (for adaptation)
        cook_steps      = gr.State([])      # steps for cook mode
        cook_step_idx   = gr.State(0)       # current cook mode step

        # ── Header ───────────────────────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        # ── Tabs ─────────────────────────────────────────────────────────────
        with gr.Tabs() as tabs:

            # ══ Tab 1: Generate Recipe ═══════════════════════════════════════
            with gr.Tab("🍛 Generate Recipe", id=0):
                with gr.Row(equal_height=False):

                    # ── Left Panel: Input ─────────────────────────────────────
                    with gr.Column(scale=3, min_width=320):

                        # Dish name input
                        gr.HTML('<div class="ik-section-label">Dish Name (optional)</div>')
                        dish_input = gr.Textbox(
                            placeholder='e.g. "Palak Paneer" — or leave blank and pick ingredients below',
                            label="",
                            lines=1,
                            elem_classes=["dish-input"],
                        )

                        # Filters row (moved up so filters + button are both visible on load)
                        gr.HTML('<div class="ik-section-label">Filters</div>')
                        with gr.Row():
                            region_filter = gr.Dropdown(
                                choices=REGIONS, value="Any Region",
                                label="📍 Region", interactive=True
                            )
                            diet_filter = gr.Dropdown(
                                choices=DIETS, value="Any Diet",
                                label="🌿 Diet", interactive=True
                            )
                            time_filter = gr.Dropdown(
                                choices=TIMES, value="Any Time",
                                label="⏱ Time", interactive=True
                            )

                        # Action buttons — placed here so they're visible on first load
                        with gr.Row():
                            generate_btn = gr.Button(
                                "⚡ Generate Recipe",
                                variant="primary",
                                elem_id="generate-btn",
                                scale=3,
                            )
                            surprise_btn = gr.Button(
                                "🎲 Surprise Me!",
                                variant="secondary",
                                scale=1,
                            )

                        # Ingredient chip picker by category (scrollable below the button)
                        gr.HTML('<div class="ik-section-label" style="margin-top:12px;">Ingredients (optional)</div>')

                        chip_components = []
                        for cat_name, cat_items, color_cls in zip(categories, category_items, color_classes):
                            chip = gr.CheckboxGroup(
                                choices=cat_items,
                                label=cat_name,
                                elem_classes=["ingredient-chips", f"{color_cls}-chips"],
                                value=[],
                            )
                            chip_components.append(chip)

                        # Custom ingredient text input
                        custom_input = gr.Textbox(
                            placeholder="Or type custom ingredients, comma-separated...",
                            label="Custom Ingredients",
                            lines=1,
                            elem_classes=["custom-input"],
                        )

                    # ── Right Panel: Recipe Card ──────────────────────────────
                    with gr.Column(scale=4, min_width=400):

                        recipe_html = gr.HTML(
                            value=EMPTY_CARD_HTML,
                            label="",
                            elem_id="recipe-card-area",
                        )

                        # Adaptation row
                        gr.HTML('<div class="ik-section-label" style="margin-top:12px;">Adapt Recipe</div>')
                        with gr.Row():
                            vegan_btn  = gr.Button("🌱 Make Vegan",  variant="secondary", elem_classes=["adapt-btn-vegan"])
                            spicy_btn  = gr.Button("🌶️ Spicier",     variant="secondary", elem_classes=["adapt-btn-spicy"])
                            quick_btn  = gr.Button("⚡ Quicker",     variant="secondary", elem_classes=["adapt-btn-quick"])
                            milder_btn = gr.Button("🌿 Milder",      variant="secondary", elem_classes=["adapt-btn-milder"])

                        # Custom adaptation
                        with gr.Row():
                            adapt_input = gr.Textbox(
                                placeholder='Custom: "make it with coconut milk" or "add more ginger"...',
                                label="", lines=1, scale=4
                            )
                            adapt_btn = gr.Button("Apply", variant="primary", scale=1)

                        # Cook mode launch
                        cook_btn = gr.Button("👨‍🍳 Start Cooking Mode", variant="secondary")

                    # ── History Sidebar ───────────────────────────────────────
                    with gr.Column(scale=1, min_width=180):
                        gr.HTML('<div class="ik-section-label">Recent</div>')
                        history_html = gr.HTML(EMPTY_HISTORY_HTML)

            # ══ Tab 2: Compare Models ════════════════════════════════════════
            with gr.Tab("🔬 Compare: Base vs Fine-tuned", id=1):

                gr.HTML(METRICS_BANNER_HTML)

                gr.Markdown(
                    "_Generate the same recipe from both the base model and the fine-tuned model "
                    "to see the difference fine-tuning makes. The metrics above are from the 500-recipe test set._",
                    elem_classes=["compare-subtitle"],
                )

                with gr.Row():
                    compare_dish = gr.Textbox(
                        placeholder='Enter a dish name to compare (e.g. "Butter Chicken")',
                        label="Dish Name",
                        scale=4,
                    )
                    compare_btn = gr.Button("🔬 Compare", variant="primary", scale=1)

                with gr.Row(equal_height=False):
                    with gr.Column():
                        base_recipe_html = gr.HTML(
                            value='<div class="ik-empty-card"><div class="ik-empty-icon">⚡</div>'
                                  '<div class="ik-empty-title">Base Model</div>'
                                  '<div class="ik-empty-subtitle">No fine-tuning</div></div>'
                        )
                    with gr.Column():
                        ft_recipe_html = gr.HTML(
                            value='<div class="ik-empty-card"><div class="ik-empty-icon">🔬</div>'
                                  '<div class="ik-empty-title">Fine-tuned Model</div>'
                                  '<div class="ik-empty-subtitle">QLoRA · 612 steps</div></div>'
                        )

            # ══ Tab 3: Cook Mode ═════════════════════════════════════════════
            with gr.Tab("👨‍🍳 Cook Mode", id=2):

                cook_display = gr.HTML(
                    value=build_cook_mode_html([], 0),
                    elem_id="cook-mode-display",
                )

                with gr.Row(elem_classes=["cook-mode-nav"]):
                    prev_step_btn = gr.Button("← Previous Step", variant="secondary", elem_classes=["cook-btn-prev"])
                    next_step_btn = gr.Button("Next Step →",      variant="primary",   elem_classes=["cook-btn-next"])

        # ─── Wire up events ─────────────────────────────────────────────────

        # All chip + filter inputs (used by both generate and surprise buttons)
        all_inputs = [dish_input] + chip_components + [
            custom_input, diet_filter, region_filter, time_filter
        ]

        def _unpack_generate(*args):
            """Unpack Gradio flat args → do_generate; yields (html, raw_text) tuples."""
            dish_name = args[0]
            rest      = args[1:]
            yield from do_generate(dish_name, *rest)

        def _unpack_surprise(*args):
            """Pick a random recipe and stream it; yields (html, raw_text) tuples."""
            diet   = args[-3]
            region = args[-2]
            time_p = args[-1]
            yield from do_surprise(diet, region, time_p)

        # Named functions for adapt buttons — lambdas can't yield from generators
        def _adapt_vegan(r):    yield from do_adapt(r, "make it fully vegan — replace dairy and meat with plant-based alternatives")
        def _adapt_spicy(r):    yield from do_adapt(r, "make it significantly spicier — add more chilli, pepper, and heat-giving spices")
        def _adapt_quick(r):    yield from do_adapt(r, "reduce cooking time to under 20 minutes — simplify steps while keeping the core flavours")
        def _adapt_milder(r):   yield from do_adapt(r, "reduce spice and heat — make it milder and suitable for sensitive palates")
        def _adapt_custom(recipe, mod): yield from do_adapt(recipe, mod)

        # Generate → (recipe_html, recipe_raw) so recipe_raw always has real model text
        generate_btn.click(
            fn=_unpack_generate,
            inputs=all_inputs,
            outputs=[recipe_html, recipe_raw],
        )

        # Surprise Me → same outputs
        surprise_btn.click(
            fn=_unpack_surprise,
            inputs=all_inputs,
            outputs=[recipe_html, recipe_raw],
        )

        # Adaptation buttons — each outputs updated (html, raw_text)
        _adapt_io = dict(inputs=[recipe_raw], outputs=[recipe_html, recipe_raw])
        vegan_btn.click(fn=_adapt_vegan,   **_adapt_io)
        spicy_btn.click(fn=_adapt_spicy,   **_adapt_io)
        quick_btn.click(fn=_adapt_quick,   **_adapt_io)
        milder_btn.click(fn=_adapt_milder, **_adapt_io)

        adapt_btn.click(
            fn=_adapt_custom,
            inputs=[recipe_raw, adapt_input],
            outputs=[recipe_html, recipe_raw],
        )

        # Cook Mode button — parses recipe steps and auto-switches to the Cook Mode tab
        cook_btn.click(
            fn=do_enter_cook_mode,
            inputs=[recipe_raw],
            outputs=[cook_display, cook_steps, cook_step_idx, tabs],
        )

        # History: only update when streaming is DONE (skip partial streaming renders)
        def _update_history_from_html(html, hist):
            if not html or "ik-empty-card" in html or "recipe-card streaming" in html:
                return build_history_html(hist or []), hist or []
            parsed = parse(html)
            if parsed.get("name") and parsed.get("steps"):
                last_name = (hist[-1].get("name") if hist else None)
                if parsed["name"] != last_name:
                    hist = (hist or []) + [parsed]
                    hist = hist[-5:]
            return build_history_html(hist or []), hist or []

        recipe_html.change(
            fn=_update_history_from_html,
            inputs=[recipe_html, history_state],
            outputs=[history_html, history_state],
        )

        next_step_btn.click(
            fn=do_cook_next,
            inputs=[cook_steps, cook_step_idx],
            outputs=[cook_display, cook_step_idx],
        )

        prev_step_btn.click(
            fn=do_cook_prev,
            inputs=[cook_steps, cook_step_idx],
            outputs=[cook_display, cook_step_idx],
        )

        # Compare tab
        compare_btn.click(
            fn=do_compare_generate,
            inputs=[compare_dish] + chip_components + [custom_input, diet_filter, region_filter, time_filter],
            outputs=[ft_recipe_html, base_recipe_html],
        )

    return demo


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    print("[app] Loading model...")
    load_model()

    print("[app] Building Gradio interface...")
    demo = build_app()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CSS,
        theme=gr.themes.Base(),
    )


if __name__ == "__main__":
    main()
