"""
Food recognition service using trained model
Replace food_recognizer.py with this file after training
"""

import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import numpy as np
from typing import List, Dict
import json
import os


class FoodRecognizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = 'models/food_model_best.pth'
        self.class_names_path = 'models/class_names.json'
        
        # Load class names
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, 'r') as f:
                self.food_classes = json.load(f)
        else:
            print("[WARNING] Class names not found, using default food list")
            self.food_classes = [
                "pizza", "burger", "sushi", "salad", "pasta",
                "chicken", "steak", "fish", "soup", "sandwich"
            ]
        
        # Load model
        self.model = None
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            print("[WARNING] Trained model not found, using mock predictions")
            print(f"[INFO] Expected model path: {os.path.abspath(self.model_path)}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.top_k = 3
    
    def _load_model(self):
        """Load trained model"""
        try:
            print(f"[INFO] Loading trained model from {self.model_path}")
            
            # Initialize model architecture
            self.model = models.efficientnet_b0(pretrained=False)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, len(self.food_classes))
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[INFO] ✓ Model loaded successfully!")
            print(f"[INFO] ✓ Classes: {len(self.food_classes)}")
            
            if 'best_val_acc' in checkpoint:
                print(f"[INFO] ✓ Model accuracy: {checkpoint['best_val_acc']:.2f}%")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None
    
    def recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Recognize food items in image using trained ML model
        
        Args:
            image: numpy array of image (RGB)
            
        Returns:
            List of food predictions with confidence scores
        """
        # Use trained model if available
        if self.model is not None:
            try:
                return self._predict_with_model(image)
            except Exception as e:
                print(f"[ERROR] Prediction failed: {e}")
                print("[INFO] Falling back to mock predictions")
                return self._mock_predictions()
        else:
            # Fallback to mock predictions
            return self._mock_predictions()
    
    def _predict_with_model(self, image: np.ndarray) -> List[Dict]:
        """Run inference with trained model"""
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top K predictions
        top_probs, top_indices = torch.topk(probabilities, min(self.top_k, len(self.food_classes)))
        
        predictions = []
        for i in range(len(top_probs)):
            idx = top_indices[i].item()
            conf = top_probs[i].item()
            
            # Get food name (clean formatting)
            food_name = self.food_classes[idx].replace('_', ' ')
            
            predictions.append({
                "name": food_name,
                "confidence": round(float(conf), 3),
                "food_id": f"food_{self.food_classes[idx]}"
            })
        
        print(f"[INFO] Predicted: {predictions[0]['name']} ({predictions[0]['confidence']:.2%})")
        
        return predictions
    
    def _mock_predictions(self) -> List[Dict]:
        """
        Generate mock predictions for development
        Used when trained model is not available
        """
        import random
        
        # Randomly select foods
        selected_foods = random.sample(self.food_classes, min(self.top_k, len(self.food_classes)))
        
        # Generate confidence scores
        confidences = sorted([random.uniform(0.3, 0.9) for _ in range(len(selected_foods))], reverse=True)
        
        predictions = []
        for food, conf in zip(selected_foods, confidences):
            food_name = food.replace('_', ' ')
            predictions.append({
                "name": food_name,
                "confidence": round(conf, 3),
                "food_id": f"food_{food}"
            })
        
        return predictions
    
    def load_model(self, model_path: str):
        """Load model from custom path"""
        self.model_path = model_path
        self._load_model()







