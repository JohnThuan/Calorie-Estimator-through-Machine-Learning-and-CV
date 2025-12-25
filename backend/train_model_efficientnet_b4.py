"""
EfficientNet-B4 Training Script with Anti-Overfitting Techniques
Target: 77-80% validation accuracy on Food-101
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATA_DIR = 'data'
    OUTPUT_DIR = 'trained_models'
    MODEL_NAME = 'efficientnet_b4_food101'
    
    # Training
    EPOCHS = 25
    BATCH_SIZE = 48  # Optimized for RTX 4070
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.01  # Strong weight decay
    
    # Model
    NUM_CLASSES = 101
    DROPOUT_RATE = 0.4  # High dropout to prevent overfitting
    
    # Augmentation
    IMAGE_SIZE = 224  # Standard size (3x faster than 380!)
    CROP_SIZE = 224
    
    # Label Smoothing
    LABEL_SMOOTHING = 0.1  # Prevents overconfident predictions
    
    # Hardware
    NUM_WORKERS = 8  # Max workers for fast data loading
    PIN_MEMORY = True
    
    # Early Stopping
    PATIENCE = 5  # Stop if val doesn't improve for 5 epochs
    MIN_DELTA = 0.001  # Minimum improvement to count


# ============================================================================
# DATA AUGMENTATION (STRONG!)
# ============================================================================

def get_transforms():
    """Strong augmentation to prevent memorization (optimized for speed)"""
    
    # Training: Strong but fast augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.CROP_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # Light random erasing
    ])
    
    # Validation: No augmentation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def create_model():
    """EfficientNet-B4 with dropout"""
    print(">> Loading EfficientNet-B4 (19M parameters)...")
    
    # Load pretrained model
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    
    # Get number of input features
    num_features = model.classifier[1].in_features
    
    # Replace classifier with dropout
    model.classifier = nn.Sequential(
        nn.Dropout(p=Config.DROPOUT_RATE, inplace=True),  # High dropout!
        nn.Linear(num_features, Config.NUM_CLASSES)
    )
    
    return model


# ============================================================================
# TRAINING LOOP
# ============================================================================

class EarlyStopping:
    """Stop training when validation loss stops improving"""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">> Device: {device}")
    
    if torch.cuda.is_available():
        print(f">> GPU: {torch.cuda.get_device_name(0)}")
        print(f">> CUDA Version: {torch.version.cuda}")
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Data
    print("\n>> Loading Food-101 dataset...")
    train_transform, val_transform = get_transforms()
    
    train_dataset = datasets.Food101(
        root=Config.DATA_DIR,
        split='train',
        transform=train_transform,
        download=False
    )
    
    val_dataset = datasets.Food101(
        root=Config.DATA_DIR,
        split='test',
        transform=val_transform,
        download=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    print(f">> Training samples: {len(train_dataset)}")
    print(f">> Validation samples: {len(val_dataset)}")
    
    # Model
    model = create_model().to(device)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print("\n" + "="*70)
    print("STARTING TRAINING - OPTIMIZED FOR SPEED")
    print("="*70)
    print(f"Model: EfficientNet-B4 (19M parameters)")
    print(f"Image Size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Workers: {Config.NUM_WORKERS}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Weight Decay: {Config.WEIGHT_DECAY}")
    print(f"Dropout: {Config.DROPOUT_RATE}")
    print(f"Label Smoothing: {Config.LABEL_SMOOTHING}")
    print(f"Target: 75-80% validation accuracy")
    print(f"Expected: ~1hr per epoch (vs 3hrs with 380px)")
    print("="*70 + "\n")
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Gap: {train_acc - val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(
                Config.OUTPUT_DIR,
                f'{Config.MODEL_NAME}_best.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, best_model_path)
            print(f"  >> BEST MODEL SAVED! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\n>> Early stopping triggered at epoch {epoch+1}")
            print(f">> Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Save final model
    final_model_path = os.path.join(
        Config.OUTPUT_DIR,
        f'{Config.MODEL_NAME}_final.pth'
    )
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'train_acc': train_acc,
    }, final_model_path)
    
    # Save training history
    history_path = os.path.join(
        Config.OUTPUT_DIR,
        f'{Config.MODEL_NAME}_history.json'
    )
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save class mapping
    class_to_idx = train_dataset.class_to_idx
    mapping_path = os.path.join(
        Config.OUTPUT_DIR,
        f'{Config.MODEL_NAME}_class_mapping.json'
    )
    with open(mapping_path, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Overfitting Gap: {history['train_acc'][-1] - history['val_acc'][-1]:.2f}%")
    print(f"\nModel saved to: {best_model_path}")
    print(f"History saved to: {history_path}")
    print(f"Class mapping saved to: {mapping_path}")
    print("="*70)


if __name__ == '__main__':
    main()

