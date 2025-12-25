"""
Food recognition using OpenAI Vision API
Fast, accurate alternative to training your own model
"""

import os
import base64
from typing import List, Dict
import numpy as np
import cv2
from openai import OpenAI


class FoodRecognizer:
    def __init__(self):
        # Set your OpenAI API key
        # Get one at: https://platform.openai.com/api-keys
        self.api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
        
        if self.api_key == 'your-api-key-here':
            print("[WARNING] OpenAI API key not set!")
            print("[INFO] Set environment variable: OPENAI_API_KEY")
            print("[INFO] Falling back to mock predictions")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.top_k = 3
    
    def recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Recognize food items using OpenAI Vision API
        """
        if self.client is None:
            return self._mock_predictions()
        
        try:
            # Convert numpy array to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this food image. Return ONLY a JSON array of the top 3 foods you see, formatted exactly like this:
[
  {"name": "pizza", "confidence": 0.95},
  {"name": "salad", "confidence": 0.80},
  {"name": "garlic bread", "confidence": 0.70}
]

Rules:
- Use lowercase food names
- Confidence between 0.0-1.0
- Most prominent food first
- Maximum 3 items"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            
            # Extract JSON from response
            import json
            import re
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                foods = json.loads(json_match.group())
                
                # Format predictions
                predictions = []
                for i, food in enumerate(foods[:self.top_k]):
                    predictions.append({
                        "name": food["name"].lower(),
                        "confidence": round(float(food["confidence"]), 3),
                        "food_id": f"food_{food['name'].lower().replace(' ', '_')}"
                    })
                
                print(f"[INFO] OpenAI recognized: {predictions[0]['name']} ({predictions[0]['confidence']:.0%})")
                return predictions
            else:
                print("[WARNING] Could not parse OpenAI response, using mock")
                return self._mock_predictions()
            
        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {e}")
            print("[INFO] Falling back to mock predictions")
            return self._mock_predictions()
    
    def _mock_predictions(self) -> List[Dict]:
        """Fallback mock predictions"""
        import random
        
        food_list = [
            "pizza", "burger", "sushi", "salad", "pasta",
            "chicken", "steak", "fish", "soup", "sandwich"
        ]
        
        selected = random.sample(food_list, self.top_k)
        confidences = sorted([random.uniform(0.3, 0.9) for _ in range(self.top_k)], reverse=True)
        
        return [
            {
                "name": food,
                "confidence": round(conf, 3),
                "food_id": f"food_{food}"
            }
            for food, conf in zip(selected, confidences)
        ]







