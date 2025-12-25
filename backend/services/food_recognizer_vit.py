"""
Food recognition using trained Vision Transformer model
Use this after training with train_model_vit.py
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
from typing import List, Dict
import json
import os
from PIL import Image


class FoodRecognizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = 'models/vit_food101/final'
        self.class_names_path = 'models/vit_food101/class_names.json'
        
        # Load class names
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, 'r') as f:
                self.food_classes = json.load(f)
        else:
            print("[WARNING] Class names not found")
            self.food_classes = []
        
        # Load model
        self.model = None
        self.processor = None
        
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            print(f"[WARNING] ViT model not found at {self.model_path}")
            print("[INFO] Using mock predictions")
        
        self.top_k = 3
    
    def _load_model(self):
        """Load trained Vision Transformer model"""
        try:
            print(f"[INFO] Loading ViT model from {self.model_path}")
            
            # Load processor
            self.processor = ViTImageProcessor.from_pretrained(self.model_path)
            
            # Load model
            self.model = ViTForImageClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[INFO] ✓ ViT model loaded successfully!")
            print(f"[INFO] ✓ Classes: {len(self.food_classes)}")
            print(f"[INFO] ✓ Device: {self.device}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load ViT model: {e}")
            self.model = None
            self.processor = None
    
    def recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Recognize food items using Vision Transformer
        
        Args:
            image: numpy array of image (RGB)
            
        Returns:
            List of food predictions with confidence scores
        """
        if self.model is not None and self.processor is not None:
            try:
                return self._predict_with_vit(image)
            except Exception as e:
                print(f"[ERROR] ViT prediction failed: {e}")
                return self._mock_predictions()
        else:
            return self._mock_predictions()
    
    def _predict_with_vit(self, image: np.ndarray) -> List[Dict]:
        """Run inference with Vision Transformer"""
        # Convert numpy to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get top K predictions
        top_probs, top_indices = torch.topk(probabilities, min(self.top_k, len(self.food_classes)))
        
        predictions = []
        for i in range(len(top_probs)):
            idx = top_indices[i].item()
            conf = top_probs[i].item()
            
            # Get food name
            food_name = self.food_classes[idx].replace('_', ' ')
            
            predictions.append({
                "name": food_name,
                "confidence": round(float(conf), 3),
                "food_id": f"food_{self.food_classes[idx]}"
            })
        
        print(f"[INFO] ViT Predicted: {predictions[0]['name']} ({predictions[0]['confidence']:.2%})")
        
        return predictions
    
    def _mock_predictions(self) -> List[Dict]:
        """Fallback mock predictions"""
        import random
        
        if not self.food_classes:
            food_list = ["pizza", "burger", "sushi", "salad", "pasta"]
        else:
            food_list = self.food_classes
        
        selected = random.sample(food_list, min(self.top_k, len(food_list)))
        confidences = sorted([random.uniform(0.3, 0.9) for _ in range(len(selected))], reverse=True)
        
        predictions = []
        for food, conf in zip(selected, confidences):
            food_name = food.replace('_', ' ')
            predictions.append({
                "name": food_name,
                "confidence": round(conf, 3),
                "food_id": f"food_{food}"
            })
        
        return predictions






