"""
Canonical ingredient list for the Indian Kitchen app.

Organized into categories that map to the chip picker UI sections.
Each category has an emoji, a color class, and a list of ingredients.

Source: top-200 most frequent ingredients in data/processed/merged_cleaned.csv,
manually curated and grouped by culinary category.
"""

INGREDIENT_CATEGORIES = {
    "🌶️ Spices & Masalas": [
        "Cumin Seeds", "Turmeric", "Coriander Powder", "Red Chilli Powder",
        "Garam Masala", "Black Pepper", "Cardamom", "Cinnamon", "Cloves",
        "Bay Leaves", "Mustard Seeds", "Fenugreek Seeds", "Kashmiri Chilli",
        "Asafoetida", "Amchur", "Chaat Masala", "Cumin Powder",
        "Fennel Seeds", "Star Anise", "Nutmeg", "Mace",
    ],
    "🧅 Aromatics": [
        "Onion", "Garlic", "Ginger", "Green Chilli", "Curry Leaves",
        "Spring Onion", "Shallots",
    ],
    "🍅 Vegetables": [
        "Tomato", "Spinach", "Potato", "Cauliflower", "Peas",
        "Eggplant", "Okra", "Bitter Gourd", "Bottle Gourd", "Drumstick",
        "Carrot", "Capsicum", "Corn", "Cabbage", "Beans", "Broccoli",
        "Mushroom", "Pumpkin", "Radish", "Beetroot", "Sweet Potato",
        "Taro Root", "Plantain", "Raw Banana",
    ],
    "🫘 Legumes & Lentils": [
        "Chickpeas", "Toor Dal", "Moong Dal", "Chana Dal", "Urad Dal",
        "Masoor Dal", "Rajma", "Black Chickpeas", "Soya Chunks",
        "Moong Beans", "Moth Beans",
    ],
    "🧀 Dairy": [
        "Paneer", "Yogurt", "Ghee", "Cream", "Butter", "Milk",
        "Khoya", "Hung Curd",
    ],
    "🍗 Proteins": [
        "Chicken", "Mutton", "Fish", "Prawns", "Eggs", "Lamb",
        "Crab", "Tofu",
    ],
    "🌾 Grains & Flour": [
        "Rice", "Basmati Rice", "Wheat Flour", "Besan", "Semolina",
        "Poha", "Vermicelli", "Oats", "Jowar Flour", "Ragi Flour",
    ],
    "🥜 Nuts & Seeds": [
        "Cashews", "Almonds", "Peanuts", "Sesame Seeds", "Poppy Seeds",
        "Melon Seeds", "Pistachios", "Walnuts",
    ],
    "🌿 Fresh Herbs": [
        "Coriander Leaves", "Mint Leaves", "Curry Leaves", "Fenugreek Leaves",
        "Basil", "Dill",
    ],
    "🫙 Pantry Staples": [
        "Oil", "Salt", "Sugar", "Tamarind", "Coconut Milk", "Coconut",
        "Tomato Puree", "Water", "Lemon", "Vinegar",
    ],
}

# Flat list for fuzzy matching in the ingredient normalizer
ALL_INGREDIENTS: list[str] = [
    ingredient
    for ingredients in INGREDIENT_CATEGORIES.values()
    for ingredient in ingredients
]

# CSS class suffix for each category (determines chip color in styles.css)
CATEGORY_COLOR_CLASS = {
    "🌶️ Spices & Masalas": "spice",
    "🧅 Aromatics": "aromatic",
    "🍅 Vegetables": "veggie",
    "🫘 Legumes & Lentils": "legume",
    "🧀 Dairy": "dairy",
    "🍗 Proteins": "protein",
    "🌾 Grains & Flour": "grain",
    "🥜 Nuts & Seeds": "nut",
    "🌿 Fresh Herbs": "herb",
    "🫙 Pantry Staples": "pantry",
}

# Region options for the filter dropdown
REGIONS = [
    "Any Region",
    "North Indian", "South Indian", "Bengali", "Gujarati",
    "Maharashtrian", "Rajasthani", "Punjabi", "Goan",
    "Kerala", "Tamil Nadu", "Andhra", "Karnataka",
    "Mughlai", "Kashmiri", "Hyderabadi", "Awadhi",
]

# Diet options
DIETS = ["Any Diet", "Vegetarian", "Vegan", "Non Vegetarian", "Jain", "Diabetic Friendly"]

# Time options
TIMES = ["Any Time", "Under 15 min", "Under 30 min", "Under 45 min", "Under 60 min", "1+ hours"]

# Quick random ingredient sets for the "Surprise Me!" button
SURPRISE_SETS = [
    ["Spinach", "Paneer", "Tomato", "Garlic", "Cumin Seeds"],
    ["Chicken", "Yogurt", "Ginger", "Garam Masala", "Onion"],
    ["Toor Dal", "Tomato", "Turmeric", "Mustard Seeds", "Curry Leaves"],
    ["Potato", "Peas", "Cumin Seeds", "Coriander Powder", "Green Chilli"],
    ["Chickpeas", "Tomato", "Onion", "Ginger", "Garlic"],
    ["Basmati Rice", "Mutton", "Yogurt", "Cardamom", "Saffron"],
    ["Cauliflower", "Potato", "Cumin Seeds", "Coriander Powder", "Turmeric"],
    ["Prawns", "Coconut Milk", "Curry Leaves", "Mustard Seeds", "Tamarind"],
    ["Moong Dal", "Spinach", "Ghee", "Turmeric", "Asafoetida"],
    ["Paneer", "Capsicum", "Tomato", "Cashews", "Cream"],
]
