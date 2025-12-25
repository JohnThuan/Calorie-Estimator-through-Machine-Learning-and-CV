"""
Food Recognition Service using EfficientNet-B4
High accuracy model with anti-overfitting techniques
Expected: 77-80% accuracy on Food-101
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import os
from typing import List, Dict

class FoodRecognizerEfficientNetB4:
    def __init__(self, model_path: str = 'trained_models/efficientnet_b4_food101_best.pth',
                 class_mapping_path: str = 'trained_models/efficientnet_b4_food101_class_mapping.json'):
        """
        Initialize the food recognizer with EfficientNet-B4
        
        Args:
            model_path: Path to the trained model checkpoint
            class_mapping_path: Path to the class mapping JSON
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.top_k = 5  # Get top 5 predictions
        self.confidence_threshold = 0.10  # Only show predictions above 10% confidence
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Food Recognizer loaded on {self.device}")
        print(f"Model: EfficientNet-B4 (19M parameters)")
        print(f"Classes: {len(self.class_to_idx)}")
    
    def _load_model(self, model_path: str):
        """Load the trained EfficientNet-B4 model"""
        # Create model architecture
        model = models.efficientnet_b4(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 101)
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Loaded model from {model_path}")
        val_acc = checkpoint.get('val_acc', 0)
        if val_acc > 0:
            print(f"Validation Accuracy: {val_acc:.2f}%")
        else:
            print(f"Validation Accuracy: N/A")
        
        return model
    
    def recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Recognize food in the image
        
        Args:
            image: Input image as numpy array (BGR from OpenCV)
            
        Returns:
            List of top-k predictions with confidence scores
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, self.top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Format results - only include predictions above confidence threshold
        predictions = []
        food_id_counter = 0
        for prob, idx in zip(top_probs, top_indices):
            # Skip predictions below confidence threshold
            if prob < self.confidence_threshold:
                continue
                
            class_name = self.idx_to_class[idx]
            # Convert food-101 format (e.g., "apple_pie") to readable format
            readable_name = class_name.replace('_', ' ').title()
            
            predictions.append({
                'food_id': f"food_{food_id_counter}",
                'name': readable_name,
                'confidence': float(prob),
                'raw_class': class_name
            })
            food_id_counter += 1
        
        # Always return at least the top prediction if nothing passed threshold
        if len(predictions) == 0:
            class_name = self.idx_to_class[top_indices[0]]
            readable_name = class_name.replace('_', ' ').title()
            predictions.append({
                'food_id': 'food_0',
                'name': readable_name,
                'confidence': float(top_probs[0]),
                'raw_class': class_name
            })
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get information about the model"""
        return {
            'architecture': 'EfficientNet-B4',
            'parameters': '19M',
            'dataset': 'Food-101',
            'classes': len(self.class_to_idx),
            'device': str(self.device),
            'expected_accuracy': '86%'
        }


# Singleton instance
_recognizer = None

def get_recognizer() -> FoodRecognizerEfficientNetB4:
    """Get or create the food recognizer instance"""
    global _recognizer
    if _recognizer is None:
        _recognizer = FoodRecognizerEfficientNetB4()
    return _recognizer


