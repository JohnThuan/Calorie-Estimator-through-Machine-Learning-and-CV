"""
Nutrition database for Food-101 dataset classes
All values are per 100g serving
Data sourced from USDA FoodData Central
"""

FOOD101_NUTRITION = {
    # Breakfast
    "french_toast": {"calories": 196, "protein": 7.7, "carbs": 28.5, "fat": 5.3},
    "pancakes": {"calories": 227, "protein": 6, "carbs": 28, "fat": 10},
    "waffles": {"calories": 291, "protein": 7.9, "carbs": 33.4, "fat": 13.6},
    "eggs_benedict": {"calories": 231, "protein": 10.3, "carbs": 8.2, "fat": 17.6},
    "omelette": {"calories": 154, "protein": 10.6, "carbs": 1.8, "fat": 11.7},
    "breakfast_burrito": {"calories": 206, "protein": 9, "carbs": 29, "fat": 6},
    
    # Appetizers & Snacks
    "spring_rolls": {"calories": 151, "protein": 4.6, "carbs": 19.7, "fat": 5.9},
    "samosa": {"calories": 262, "protein": 5.4, "carbs": 28.8, "fat": 13.9},
    "edamame": {"calories": 121, "protein": 11.9, "carbs": 8.9, "fat": 5.2},
    "deviled_eggs": {"calories": 152, "protein": 12.6, "carbs": 0.8, "fat": 10.5},
    "chicken_wings": {"calories": 203, "protein": 30.5, "carbs": 0, "fat": 8.1},
    "oysters": {"calories": 68, "protein": 7.1, "carbs": 3.9, "fat": 2.5},
    "fried_calamari": {"calories": 175, "protein": 13.2, "carbs": 9.8, "fat": 9.4},
    "mussels": {"calories": 86, "protein": 11.9, "carbs": 3.7, "fat": 2.2},
    
    # Salads
    "caesar_salad": {"calories": 190, "protein": 3.4, "carbs": 8.6, "fat": 16.4},
    "greek_salad": {"calories": 106, "protein": 3.9, "carbs": 5.4, "fat": 7.9},
    "caprese_salad": {"calories": 160, "protein": 10.2, "carbs": 3.8, "fat": 11.6},
    "cobb_salad": {"calories": 104, "protein": 8.4, "carbs": 3.8, "fat": 6.4},
    "seaweed_salad": {"calories": 70, "protein": 1.5, "carbs": 12.9, "fat": 1.5},
    
    # Soups
    "miso_soup": {"calories": 40, "protein": 2.2, "carbs": 5.3, "fat": 1.2},
    "french_onion_soup": {"calories": 56, "protein": 3.7, "carbs": 7.8, "fat": 1.7},
    "clam_chowder": {"calories": 72, "protein": 4.6, "carbs": 9.1, "fat": 2.0},
    "hot_and_sour_soup": {"calories": 45, "protein": 3.2, "carbs": 5.4, "fat": 1.5},
    
    # Sandwiches & Burgers
    "hamburger": {"calories": 295, "protein": 17, "carbs": 24, "fat": 14},
    "hot_dog": {"calories": 290, "protein": 10.4, "carbs": 24.3, "fat": 17.6},
    "club_sandwich": {"calories": 238, "protein": 16.2, "carbs": 20.8, "fat": 10.4},
    "pulled_pork_sandwich": {"calories": 233, "protein": 18.2, "carbs": 22.5, "fat": 7.8},
    "reuben_sandwich": {"calories": 275, "protein": 15.6, "carbs": 20.3, "fat": 14.2},
    "pastrami_sandwich": {"calories": 262, "protein": 17.8, "carbs": 24.6, "fat": 10.9},
    "grilled_cheese_sandwich": {"calories": 312, "protein": 12.4, "carbs": 28.6, "fat": 16.4},
    
    # Pizza & Italian
    "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fat": 10},
    "spaghetti_carbonara": {"calories": 195, "protein": 7.8, "carbs": 24.9, "fat": 7.3},
    "spaghetti_bolognese": {"calories": 150, "protein": 8.2, "carbs": 20.6, "fat": 3.9},
    "ravioli": {"calories": 175, "protein": 6.6, "carbs": 29.3, "fat": 3.5},
    "lasagna": {"calories": 135, "protein": 8.1, "carbs": 13.2, "fat": 5.6},
    "risotto": {"calories": 143, "protein": 3.8, "carbs": 20.5, "fat": 4.8},
    "gnocchi": {"calories": 150, "protein": 3.5, "carbs": 32.0, "fat": 0.3},
    
    # Asian Cuisine
    "sushi": {"calories": 143, "protein": 6, "carbs": 21, "fat": 4},
    "sashimi": {"calories": 127, "protein": 20.5, "carbs": 0, "fat": 4.9},
    "ramen": {"calories": 436, "protein": 14.3, "carbs": 62.4, "fat": 14.5},
    "pho": {"calories": 194, "protein": 9.2, "carbs": 29.2, "fat": 3.5},
    "pad_thai": {"calories": 354, "protein": 9.9, "carbs": 41.9, "fat": 16.4},
    "dumplings": {"calories": 206, "protein": 8.3, "carbs": 24.7, "fat": 8.2},
    "gyoza": {"calories": 179, "protein": 7.9, "carbs": 20.5, "fat": 7.4},
    "takoyaki": {"calories": 150, "protein": 5.2, "carbs": 15.8, "fat": 7.6},
    "bibimbap": {"calories": 145, "protein": 6.8, "carbs": 21.7, "fat": 3.2},
    "fried_rice": {"calories": 163, "protein": 4.0, "carbs": 28.7, "fat": 3.0},
    
    # Mexican
    "tacos": {"calories": 226, "protein": 9.5, "carbs": 20.8, "fat": 11.9},
    "burritos": {"calories": 206, "protein": 9, "carbs": 29, "fat": 6},
    "nachos": {"calories": 346, "protein": 9.8, "carbs": 36.0, "fat": 18.9},
    "quesadilla": {"calories": 234, "protein": 10.4, "carbs": 22.7, "fat": 11.4},
    "guacamole": {"calories": 160, "protein": 2.0, "carbs": 8.5, "fat": 14.7},
    "ceviche": {"calories": 87, "protein": 13.2, "carbs": 6.8, "fat": 0.9},
    
    # Indian
    "chicken_curry": {"calories": 119, "protein": 11.3, "carbs": 6.2, "fat": 5.6},
    "tandoori_chicken": {"calories": 148, "protein": 22.7, "carbs": 2.4, "fat": 5.1},
    
    # Meats & Proteins
    "steak": {"calories": 271, "protein": 26, "carbs": 0, "fat": 19},
    "prime_rib": {"calories": 287, "protein": 24.5, "carbs": 0, "fat": 20.3},
    "pork_chop": {"calories": 231, "protein": 25.7, "carbs": 0, "fat": 13.9},
    "baby_back_ribs": {"calories": 290, "protein": 25.8, "carbs": 0, "fat": 20.5},
    "beef_carpaccio": {"calories": 120, "protein": 21.8, "carbs": 0.2, "fat": 3.0},
    "beef_tartare": {"calories": 135, "protein": 20.1, "carbs": 0.5, "fat": 5.6},
    "foie_gras": {"calories": 462, "protein": 11.4, "carbs": 4.7, "fat": 43.8},
    
    # Poultry
    "chicken_quesadilla": {"calories": 234, "protein": 10.4, "carbs": 22.7, "fat": 11.4},
    "peking_duck": {"calories": 337, "protein": 18.6, "carbs": 0.4, "fat": 28.4},
    
    # Seafood
    "fish_and_chips": {"calories": 232, "protein": 11.6, "carbs": 18.5, "fat": 13.0},
    "grilled_salmon": {"calories": 206, "protein": 22, "carbs": 0, "fat": 12},
    "tuna_tartare": {"calories": 108, "protein": 23.3, "carbs": 0, "fat": 0.5},
    "lobster_roll_sandwich": {"calories": 258, "protein": 17.3, "carbs": 23.6, "fat": 11.2},
    "lobster_bisque": {"calories": 92, "protein": 4.3, "carbs": 7.6, "fat": 4.9},
    "shrimp_and_grits": {"calories": 142, "protein": 9.4, "carbs": 15.2, "fat": 5.1},
    "scallops": {"calories": 111, "protein": 20.5, "carbs": 5.4, "fat": 0.8},
    "escargots": {"calories": 90, "protein": 16.5, "carbs": 2.0, "fat": 1.4},
    
    # Pasta & Noodles
    "macaroni_and_cheese": {"calories": 164, "protein": 6.4, "carbs": 18.8, "fat": 6.6},
    "fettuccine_alfredo": {"calories": 161, "protein": 5.9, "carbs": 19.8, "fat": 6.5},
    "poutine": {"calories": 510, "protein": 15.2, "carbs": 58.3, "fat": 24.5},
    
    # Sides
    "french_fries": {"calories": 312, "protein": 3.4, "carbs": 41.4, "fat": 14.7},
    "onion_rings": {"calories": 411, "protein": 5.3, "carbs": 38.2, "fat": 26.6},
    "tater_tots": {"calories": 244, "protein": 3.1, "carbs": 31.1, "fat": 11.9},
    "garlic_bread": {"calories": 350, "protein": 9.8, "carbs": 43.2, "fat": 15.6},
    "hummus": {"calories": 166, "protein": 7.9, "carbs": 14.3, "fat": 9.6},
    "bruschetta": {"calories": 106, "protein": 3.6, "carbs": 16.4, "fat": 3.1},
    
    # Desserts
    "ice_cream": {"calories": 207, "protein": 3.5, "carbs": 23.6, "fat": 11.0},
    "chocolate_cake": {"calories": 371, "protein": 4.9, "carbs": 50.7, "fat": 16.9},
    "cheesecake": {"calories": 321, "protein": 5.5, "carbs": 25.5, "fat": 22.5},
    "apple_pie": {"calories": 237, "protein": 2.4, "carbs": 34.3, "fat": 11.0},
    "carrot_cake": {"calories": 350, "protein": 3.9, "carbs": 48.2, "fat": 16.5},
    "red_velvet_cake": {"calories": 385, "protein": 4.2, "carbs": 51.8, "fat": 18.4},
    "tiramisu": {"calories": 240, "protein": 4.8, "carbs": 28.5, "fat": 11.6},
    "churros": {"calories": 340, "protein": 4.7, "carbs": 40.2, "fat": 17.8},
    "macarons": {"calories": 380, "protein": 5.7, "carbs": 54.2, "fat": 16.4},
    "creme_brulee": {"calories": 261, "protein": 4.8, "carbs": 25.5, "fat": 15.3},
    "panna_cotta": {"calories": 242, "protein": 3.2, "carbs": 16.8, "fat": 18.5},
    "chocolate_mousse": {"calories": 302, "protein": 4.3, "carbs": 25.6, "fat": 20.8},
    "baklava": {"calories": 428, "protein": 5.9, "carbs": 50.7, "fat": 23.0},
    "strawberry_shortcake": {"calories": 225, "protein": 3.2, "carbs": 32.5, "fat": 9.4},
    "beignets": {"calories": 349, "protein": 6.9, "carbs": 43.2, "fat": 16.8},
    "donuts": {"calories": 452, "protein": 5.0, "carbs": 50.8, "fat": 25.5},
    "cupcakes": {"calories": 305, "protein": 3.5, "carbs": 44.2, "fat": 13.1},
    "bread_pudding": {"calories": 214, "protein": 6.1, "carbs": 31.2, "fat": 7.6},
    "frozen_yogurt": {"calories": 127, "protein": 3.5, "carbs": 23.9, "fat": 2.0},
    
    # Drinks & Misc
    "cup_cakes": {"calories": 305, "protein": 3.5, "carbs": 44.2, "fat": 13.1},
}


def get_nutrition(food_name: str) -> dict:
    """
    Get nutrition info for a food item
    
    Args:
        food_name: Name of the food (lowercase with underscores)
    
    Returns:
        Dictionary with calories, protein, carbs, and fat per 100g
    """
    # Normalize food name
    normalized = food_name.lower().replace(' ', '_')
    
    # Return from database or default
    if normalized in FOOD101_NUTRITION:
        return FOOD101_NUTRITION[normalized]
    else:
        # Default for unknown foods (generic meal)
        return {"calories": 180, "protein": 8, "carbs": 22, "fat": 7}


