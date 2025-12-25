"""
Calorie calculation service
Maps food items and portions to nutrition data
"""

from typing import List, Dict
import json
import os
from .food101_nutrition import get_nutrition


class CalorieCalculator:
    def __init__(self):
        # Using comprehensive Food-101 nutrition database
        pass
    
    def calculate_ranges(
        self,
        food_predictions: List[Dict],
        portions: List[Dict]
    ) -> Dict:
        """
        Calculate calorie ranges with confidence intervals
        
        Returns dictionary with min, max, estimate, and confidence
        """
        total_calories_min = 0
        total_calories_max = 0
        total_calories_estimate = 0
        
        nutrients = {"protein": 0, "carbs": 0, "fat": 0}
        
        confidence_weights = []
        
        for food, portion in zip(food_predictions, portions):
            # Get raw class name (with underscores) for database lookup
            food_key = food.get("raw_class", food["name"].lower().replace(' ', '_'))
            grams = portion["grams"]
            confidence = min(food["confidence"], portion["confidence"])
            
            # Get nutrition data from comprehensive database
            nutrition = get_nutrition(food_key)
            
            # Calculate calories for this portion
            cal_per_gram = nutrition["calories"] / 100
            estimated_calories = cal_per_gram * grams
            
            # Add uncertainty range based on confidence
            uncertainty = (1 - confidence) * 0.3  # Up to 30% uncertainty
            min_cal = estimated_calories * (1 - uncertainty)
            max_cal = estimated_calories * (1 + uncertainty)
            
            total_calories_min += min_cal
            total_calories_max += max_cal
            total_calories_estimate += estimated_calories
            
            # Calculate nutrients
            for nutrient in ["protein", "carbs", "fat"]:
                nutrient_per_gram = nutrition[nutrient] / 100
                nutrients[nutrient] += nutrient_per_gram * grams
            
            confidence_weights.append(confidence)
        
        # Overall confidence is weighted average
        overall_confidence = sum(confidence_weights) / len(confidence_weights) if confidence_weights else 0.5
        
        return {
            "min": int(total_calories_min),
            "max": int(total_calories_max),
            "estimate": int(total_calories_estimate),
            "confidence": round(overall_confidence, 3),
            "nutrients": {
                "protein": round(nutrients["protein"], 1),
                "carbs": round(nutrients["carbs"], 1),
                "fat": round(nutrients["fat"], 1)
            }
        }
    
    def adjust_for_user_feedback(
        self,
        food_id: str,
        reported_calories: int,
        estimated_calories: int
    ):
        """
        Adjust calibration based on user feedback
        """
        # TODO: Implement calibration adjustment
        # Store correction factor for this food
        pass


