"""
Create class mapping JSON for Food-101 dataset
"""
from torchvision.datasets import Food101
import json
import os

# Load dataset to get class mapping
dataset = Food101(root='data', split='train', download=False)
mapping = dataset.class_to_idx

# Save to JSON
os.makedirs('trained_models', exist_ok=True)
output_path = 'trained_models/efficientnet_b4_food101_class_mapping.json'

with open(output_path, 'w') as f:
    json.dump(mapping, f, indent=2)

print('Class mapping created successfully!')
print(f'Saved to: {output_path}')
print(f'Total classes: {len(mapping)}')


