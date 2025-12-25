"""
Image processing service
Handles image preprocessing and portion estimation
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Dict


class ImageProcessor:
    def __init__(self):
        self.target_size = (512, 512)
        self.reference_object_size = 100  # pixels for reference
    
    def process_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Process uploaded image for analysis
        - Resize to standard dimensions
        - Normalize
        - Apply preprocessing
        """
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        return img_array
    
    def estimate_portions(
        self,
        image: np.ndarray,
        food_predictions: List[Dict]
    ) -> List[Dict]:
        """
        Estimate portion sizes using image heuristics
        
        Uses segmentation and area analysis to estimate serving sizes
        """
        portions = []
        
        # Convert to grayscale for segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Calculate total food area
        total_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100)
        
        # Estimate portions for each food
        for i, food in enumerate(food_predictions):
            # Heuristic: distribute area proportionally to confidence
            food_area = total_area * (food["confidence"] / sum(f["confidence"] for f in food_predictions))
            
            # Convert area to portion estimate
            portion_size, grams = self._area_to_portion(food_area, food["name"])
            
            portions.append({
                "food_id": food.get("food_id", f"food_{i}"),
                "portion_size": portion_size,
                "grams": grams,
                "confidence": food["confidence"] * 0.7  # Reduce confidence for portion estimation
            })
        
        return portions
    
    def _area_to_portion(self, area: float, food_name: str) -> tuple:
        """
        Convert pixel area to portion size estimate
        
        Returns: (portion_size_label, grams_estimate)
        """
        # Calibrated portion estimates (reduced by ~40-50%)
        # These are more realistic serving sizes
        
        if area < 5000:
            return "small", 60.0  # ~2 oz
        elif area < 15000:
            return "medium", 120.0  # ~4 oz
        elif area < 30000:
            return "large", 200.0  # ~7 oz
        else:
            return "extra_large", 300.0  # ~10 oz
    
    def detect_reference_object(self, image: np.ndarray) -> float:
        """
        Detect reference object (e.g., credit card, coin) for scale calibration
        
        Returns scale factor
        """
        # Placeholder for reference object detection
        # Could use color detection, shape detection, or marker detection
        return 1.0


