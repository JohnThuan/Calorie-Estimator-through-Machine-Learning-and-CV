"""
Train Food Recognition Model with Vision Transformer (ViT)
Target: 96% accuracy like the Hugging Face model
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer
)
from torchvision import datasets
from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm

print("=" * 60)
print(">>> Training with Vision Transformer (Target: 96% accuracy)")
print("=" * 60)

# Configuration (memory-optimized for RTX 4070)
CONFIG = {
    'model_name': 'google/vit-base-patch16-224-in21k',
    'data_dir': 'data/food-101/images',
    'output_dir': 'models/vit_food101',
    'batch_size': 32,  # Reduced for memory efficiency
    'gradient_accumulation_steps': 16,  # 32 * 16 = 512 effective batch
    'learning_rate': 0.005,
    'epochs': 5,
    'eval_steps': 500,
    'save_steps': 500,
    'warmup_ratio': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dataloader_num_workers': 0,  # Single-threaded for memory efficiency
}

print(f"Device: {CONFIG['device']}")
print(f"Effective Batch Size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"Epochs: {CONFIG['epochs']}")
print(f"Learning Rate: {CONFIG['learning_rate']}")
print("=" * 60)

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Load image processor
print("\n>> Loading ViT image processor...")
processor = ViTImageProcessor.from_pretrained(CONFIG['model_name'])

# Load dataset
print(">> Loading Food-101 dataset...")
full_dataset = datasets.ImageFolder(root=CONFIG['data_dir'])

# Get class names
class_names = full_dataset.classes
id2label = {i: label for i, label in enumerate(class_names)}
label2id = {label: i for i, label in enumerate(class_names)}

# Save class names
with open(os.path.join(CONFIG['output_dir'], 'class_names.json'), 'w') as f:
    json.dump(class_names, f, indent=2)

print(f">> Classes: {len(class_names)}")

# Split dataset (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f">> Training samples: {train_size}")
print(f">> Validation samples: {val_size}")

# Create custom collator for memory-efficient loading
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator

class ImageCollator:
    """Memory-efficient image collator that processes images on-the-fly"""
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images = [item[0].convert('RGB') for item in batch]
        labels = [item[1] for item in batch]
        
        # Process images
        encoding = self.processor(images, return_tensors='pt')
        encoding['labels'] = torch.tensor(labels)
        
        return encoding

# Create data collator
print("\n>> Setting up memory-efficient data loading...")
collator = ImageCollator(processor)

# Load ViT model
print("\n>> Loading Vision Transformer model...")
model = ViTForImageClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=len(class_names),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

print(f">> Model: Vision Transformer Base")
print(f">> Parameters: ~86M")

# Training arguments (memory-optimized)
training_args = TrainingArguments(
    output_dir=CONFIG['output_dir'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['batch_size'],
    gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
    evaluation_strategy="steps",
    eval_steps=CONFIG['eval_steps'],
    save_strategy="steps",
    save_steps=CONFIG['save_steps'],
    learning_rate=CONFIG['learning_rate'],
    num_train_epochs=CONFIG['epochs'],
    warmup_ratio=CONFIG['warmup_ratio'],
    logging_dir=os.path.join(CONFIG['output_dir'], 'logs'),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    fp16=True if CONFIG['device'] == 'cuda' else False,  # Mixed precision
    dataloader_num_workers=CONFIG['dataloader_num_workers'],
    dataloader_pin_memory=True,
)

# Metrics
def compute_metrics(eval_pred):
    """Compute accuracy"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions.predictions if hasattr(predictions, 'predictions') else predictions, axis=1)
    labels = labels.label_ids if hasattr(labels, 'label_ids') else labels
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

# Create Trainer
print("\n>> Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# Train!
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(">>> STARTING TRAINING")
    print("=" * 60)
    print("Target: 96% validation accuracy")
    print("Expected time: 1-2 hours with RTX 4070")
    print("=" * 60 + "\n")
    
    # Train
    train_result = trainer.train()
    
    # Evaluate
    print("\n>> Final Evaluation...")
    eval_result = trainer.evaluate()
    
    # Save final model
    print("\n>> Saving model...")
    trainer.save_model(os.path.join(CONFIG['output_dir'], 'final'))
    processor.save_pretrained(os.path.join(CONFIG['output_dir'], 'final'))
    
    # Print results
    print("\n" + "=" * 60)
    print(">>> TRAINING COMPLETE!")
    print("=" * 60)
    print(f">> Final Accuracy: {eval_result['eval_accuracy']:.4f} ({eval_result['eval_accuracy']*100:.2f}%)")
    print(f">> Final Loss: {eval_result['eval_loss']:.4f}")
    print(f">> Model saved to: {CONFIG['output_dir']}/final")
    print("=" * 60)
    
    # Compare to target
    target_accuracy = 0.96
    achieved = eval_result['eval_accuracy']
    
    if achieved >= target_accuracy:
        print(f"\n>> SUCCESS! Achieved {achieved*100:.2f}% (Target: {target_accuracy*100}%)")
    elif achieved >= 0.90:
        print(f"\n>> EXCELLENT! Achieved {achieved*100:.2f}% (Close to target {target_accuracy*100}%)")
    elif achieved >= 0.85:
        print(f"\n>>> VERY GOOD! Achieved {achieved*100:.2f}% (Target: {target_accuracy*100}%)")
    else:
        print(f"\n>> Achieved {achieved*100:.2f}% (Target: {target_accuracy*100}%)")
    
    print("\n>> Next: Update backend to use this model!")
    print("   Run: python update_to_vit.py")

