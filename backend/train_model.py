"""
Train Food Recognition Model on Food-101 Dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import json
from datetime import datetime

# Configuration
CONFIG = {
    'data_dir': 'data/food-101/images',
    'model_save_dir': 'models',
    'batch_size': 64,  # Optimized for RTX 4070 (12GB VRAM)
    'epochs': 15,
    'learning_rate': 0.001,
    'num_workers': 8,  # More workers for faster data loading
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_every': 5,  # Save checkpoint every N epochs
}

print("=" * 60)
print("🍕 NutriLens Food Recognition Model Training")
print("=" * 60)
print(f"Device: {CONFIG['device']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Epochs: {CONFIG['epochs']}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print("=" * 60)

# Create model directory
os.makedirs(CONFIG['model_save_dir'], exist_ok=True)

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\n📂 Loading datasets...")

# Load datasets
train_dataset = datasets.ImageFolder(
    root=os.path.join(CONFIG['data_dir']),
    transform=train_transform
)

# Split dataset (Food-101 has predefined train/test split in meta folder)
# For simplicity, we'll do 80/20 split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Update validation transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True, 
    num_workers=CONFIG['num_workers'],
    pin_memory=True if CONFIG['device'] == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=CONFIG['num_workers'],
    pin_memory=True if CONFIG['device'] == 'cuda' else False
)

print(f"✓ Training samples: {train_size}")
print(f"✓ Validation samples: {val_size}")
print(f"✓ Number of classes: {len(train_dataset.dataset.classes)}")

# Save class names
class_names = train_dataset.dataset.classes
with open(os.path.join(CONFIG['model_save_dir'], 'class_names.json'), 'w') as f:
    json.dump(class_names, f, indent=2)
print(f"✓ Class names saved to {CONFIG['model_save_dir']}/class_names.json")

# Initialize model
print("\n🔧 Initializing model...")
model = models.efficientnet_b0(pretrained=True)

# Freeze early layers for faster training
for param in list(model.parameters())[:-20]:
    param.requires_grad = False

# Modify classifier for 101 classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(class_names))
model = model.to(CONFIG['device'])

print(f"✓ Model: EfficientNet-B0")
print(f"✓ Output classes: {len(class_names)}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                  factor=0.5, patience=2, verbose=True)

# TensorBoard
writer = SummaryWriter(f'runs/food101_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Val]  ')
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# Training loop
if __name__ == '__main__':
    print("\nStarting training...")
    print("=" * 60)

    best_val_acc = 0.0

    for epoch in range(CONFIG['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device'], epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, CONFIG['device'], epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{CONFIG["epochs"]} Summary:')
        print(f'   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'   Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % CONFIG['save_every'] == 0:
            checkpoint_path = os.path.join(
                CONFIG['model_save_dir'], 
                f'food_model_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'   [SAVED] Checkpoint: {checkpoint_path}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(CONFIG['model_save_dir'], 'food_model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f'   [BEST] Model saved! Val Acc: {val_acc:.2f}%')
    
        print("=" * 60)

    # Save final model
    final_model_path = os.path.join(CONFIG['model_save_dir'], 'food_model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'config': CONFIG,
        'final_train_acc': train_acc,
        'final_val_acc': val_acc,
        'best_val_acc': best_val_acc,
    }, final_model_path)

    writer.close()

    print("\n" + "=" * 60)
    print("[SUCCESS] Training Complete!")
    print("=" * 60)
    print(f"Final Training Accuracy: {train_acc:.2f}%")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {final_model_path}")
    print(f"TensorBoard logs: runs/")
    print("\nNext step: Update backend to use trained model!")
    print("   Run: python update_recognizer.py")
    print("=" * 60)

