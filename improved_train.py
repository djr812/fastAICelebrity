#!/usr/bin/env python
"""
Improved Celebrity Recognition Training Script
- Implements more robust training techniques
- Better regularization to prevent overfitting
- More extensive data augmentation
- Class balancing to handle imbalanced data
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import argparse
import random
from tqdm import tqdm

# Configuration
DATA_DIR = Path("img_align_celeba")
IDENTITY_FILE = "list_identity_celeba.txt"
MODEL_PATH = Path("models")
MODEL_NAME = "improved_celebrity_model.pth"
LOG_DIR = Path("logs")

# Training parameters that address overfitting
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-5
DROPOUT_PROB = 0.3
NUM_CLASSES = 100
MIN_SAMPLES_PER_CLASS = 10
USE_MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 3

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Improved Celebrity Recognition Training")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, 
                        help=f"Batch size for training (default: {BATCH_SIZE})")
    parser.add_argument("--epochs", type=int, default=EPOCHS, 
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--lr", type=float, default=LR, 
                        help=f"Learning rate (default: {LR})")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, 
                        help=f"Input image size (default: {IMAGE_SIZE})")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, 
                        help=f"Number of classes to train (default: {NUM_CLASSES})")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_CLASS, 
                        help=f"Minimum samples per class (default: {MIN_SAMPLES_PER_CLASS})")
    parser.add_argument("--model", type=str, default="resnet50", 
                        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"], 
                        help="Model architecture to use (default: resnet50)")
    parser.add_argument("--no-cuda", action="store_true", 
                        help="Disable CUDA even if available")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    return parser.parse_args()

def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def load_identity_data(min_samples=MIN_SAMPLES_PER_CLASS, num_classes=NUM_CLASSES):
    """Load identity data with improved filtering"""
    print("Loading celebrity identity data...")
    
    # Load the identity file
    try:
        df = pd.read_csv(IDENTITY_FILE, sep="\s+", skiprows=1, 
                         names=["image_id", "identity_name"])
        print(f"Loaded {len(df)} total images")
    except Exception as e:
        print(f"Error loading identity file: {e}")
        print("Using random data for demonstration...")
        # Create dummy data for testing
        df = pd.DataFrame({
            "image_id": [f"{i:06d}.jpg" for i in range(1000)],
            "identity_name": np.random.randint(0, 100, 1000)
        })
    
    # Count samples per class and filter classes with enough samples
    class_counts = df["identity_name"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index.tolist()
    
    # Limit to top N most frequent classes
    valid_classes = valid_classes[:num_classes]
    print(f"Selected {len(valid_classes)} classes with at least {min_samples} samples each")
    
    # Filter the dataframe to only include these classes
    filtered_df = df[df["identity_name"].isin(valid_classes)].copy()
    print(f"Working with {len(filtered_df)} images")
    
    # Create a mapping from identity name to class index
    class_to_idx = {identity: idx for idx, identity in enumerate(valid_classes)}
    filtered_df["class_idx"] = filtered_df["identity_name"].map(class_to_idx)
    
    # Save class mapping for later use
    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(MODEL_PATH/"class_mapping.txt", "w") as f:
        for identity, idx in class_to_idx.items():
            f.write(f"{identity},{idx}\n")
    
    # Create a mapping file with real names
    create_celebrity_names(valid_classes)
    
    return filtered_df, len(valid_classes)

def create_celebrity_names(class_ids):
    """Create a mapping file with celebrity names"""
    # Read the celebrity names file if it exists
    celeb_names = []
    if os.path.exists("celebrities.txt"):
        with open("celebrities.txt", "r") as f:
            celeb_names = [line.strip() for line in f.readlines() if line.strip()]
    
    # Make sure we have enough names
    while len(celeb_names) < len(class_ids):
        celeb_names.append(f"Celebrity_{len(celeb_names)}")
    
    # Save the mapping
    with open(MODEL_PATH/"celebrity_names.txt", "w") as f:
        for i, class_id in enumerate(class_ids):
            if i < len(celeb_names):
                f.write(f"{class_id},{celeb_names[i]}\n")
            else:
                f.write(f"{class_id},Celebrity_{class_id}\n")
    
    print(f"Created celebrity name mapping with {len(class_ids)} entries")

class CelebDataset(Dataset):
    """Improved celebrity dataset with data augmentation"""
    def __init__(self, df, root_dir, transform=None, is_train=True):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a solid color image as fallback
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(100, 100, 100))
            
        label = self.df.iloc[idx]["class_idx"]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(image_size):
    """Create separate transforms for training and validation"""
    # Stronger augmentation for training to prevent overfitting
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])
    
    # Simpler transforms for validation
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

def create_model(num_classes, model_name="resnet50", dropout_prob=DROPOUT_PROB):
    """Create model with improved regularization"""
    print(f"Creating {model_name} model with {num_classes} output classes")
    
    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "resnet34":
        model = models.resnet34(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "efficientnet_b0":
        # Try to load EfficientNet
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout_prob, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        except ImportError:
            print("EfficientNet not available, falling back to ResNet50")
            model_name = "resnet50"
    
    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, num_classes)
        )
    
    return model

def compute_balanced_loss_weights(labels):
    """Compute balanced class weights for loss function"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights)

def train_model(model, train_loader, val_loader, 
                epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Train the model with improved regularization and monitoring"""
    model = model.to(device)
    
    # Get class weights for balanced loss
    train_labels = [label for _, label in train_loader.dataset]
    class_weights = compute_balanced_loss_weights(train_labels).to(device)
    
    # Create loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Adam optimizer with weight decay regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Initialize mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION and torch.cuda.is_available() else None
    
    # Training records
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Directory for saving logs and models
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Calculate metrics
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item() * inputs.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * train_correct / train_total
            })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate metrics
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item() * inputs.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100 * val_correct / val_total
                })
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            print(f"  Saving best model (val_loss: {val_loss:.4f})")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'num_classes': model.fc[1].out_features if isinstance(model.fc, nn.Sequential) else model.fc.out_features
            }, MODEL_PATH/MODEL_NAME)
        else:
            patience_counter += 1
            
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'num_classes': model.fc[1].out_features if isinstance(model.fc, nn.Sequential) else model.fc.out_features
            }, MODEL_PATH/f"checkpoint_epoch{epoch+1}.pth")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot and save training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(LOG_DIR/'training_history.png')
    plt.close()
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Load best model for final metrics
    checkpoint = torch.load(MODEL_PATH/MODEL_NAME)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    set_seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Load identity data
    df, num_classes = load_identity_data(
        min_samples=args.min_samples,
        num_classes=args.num_classes
    )
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["class_idx"], random_state=args.seed
    )
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    # Get data transforms
    train_transform, val_transform = get_transforms(args.image_size)
    
    # Create datasets
    train_dataset = CelebDataset(train_df, DATA_DIR, transform=train_transform, is_train=True)
    val_dataset = CelebDataset(val_df, DATA_DIR, transform=val_transform, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Create model
    model = create_model(
        num_classes=num_classes,
        model_name=args.model,
        dropout_prob=DROPOUT_PROB
    )
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        device=device
    )
    
    print("Training completed successfully!")
    print(f"Model saved at {MODEL_PATH/MODEL_NAME}")
    
    return 0

if __name__ == "__main__":
    main() 