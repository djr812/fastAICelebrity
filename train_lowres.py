#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch
from fastai.vision.all import *
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse

# Configuration
DATA_DIR = Path("img_align_celeba")
IDENTITY_FILE = "list_identity_celeba.txt"
MODEL_PATH = Path("models")
MODEL_NAME = "celebrity_recognition_model.pkl"
BATCH_SIZE = 8  # Very small batch size for limited memory
IMAGE_SIZE = 128  # Small image size for limited memory
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
MAX_CELEBRITIES = 100  # Limit number of celebrities to reduce memory needs

def load_identity_data(max_celebs=MAX_CELEBRITIES):
    """Load celebrity identity data from the text file, limiting to top celebrities."""
    print("Loading identity data...")
    
    # Skip the first few lines which contain metadata
    df = pd.read_csv(IDENTITY_FILE, sep="\s+", skiprows=1, 
                     names=["image_id", "identity_name"])
    
    print(f"Loaded {len(df)} image identity records")
    
    # Get most frequent celebrities to limit dataset size
    if max_celebs and max_celebs > 0:
        top_celebs = df['identity_name'].value_counts().head(max_celebs).index
        df = df[df['identity_name'].isin(top_celebs)].copy()
        print(f"Limited dataset to top {max_celebs} celebrities ({len(df)} images)")
    
    return df

def setup_data_loaders(df, device='cuda'):
    """Set up FastAI data loaders for training with memory optimization."""
    # Make sure model directory exists
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Filter out celebrities with fewer than 2 images to avoid stratification issues
    celebrity_counts = df['identity_name'].value_counts()
    valid_celebrities = celebrity_counts[celebrity_counts >= 2].index
    filtered_df = df[df['identity_name'].isin(valid_celebrities)].copy()
    
    print(f"Filtered out {len(df) - len(filtered_df)} images with too few examples per celebrity")
    print(f"Proceeding with {len(filtered_df)} images of {len(valid_celebrities)} celebrities")
    
    # Split data into train and validation sets
    train_df, valid_df = train_test_split(
        filtered_df, test_size=0.2, stratify=filtered_df['identity_name'], random_state=42
    )
    
    # Reset indices for cleaner dataframes
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(valid_df)} images")
    
    # Create FastAI DataBlock with minimal transforms to save memory
    celeb_data = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=lambda r: DATA_DIR/r['image_id'],
        get_y=lambda r: r['identity_name'],
        splitter=ColSplitter(),
        item_tfms=Resize(IMAGE_SIZE),
        batch_tfms=Normalize.from_stats(*imagenet_stats)  # Minimal augmentation
    )
    
    # Add a column for splitting the data
    filtered_df['is_valid'] = np.where(filtered_df['image_id'].isin(valid_df['image_id']), True, False)
    
    # Create dataloaders with small batch size
    dls = celeb_data.dataloaders(filtered_df, bs=BATCH_SIZE, device=device)
    return dls

def train_model(dls, epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """Train the celebrity recognition model with memory optimizations."""
    print("\nTraining model...")
    
    # Use a smaller architecture
    model_arch = resnet18  # Much smaller than resnet50
    
    # Create a CNN learner
    learn = vision_learner(dls, model_arch, metrics=[error_rate, accuracy])
    
    # Train first the head to save memory during initial training
    print("Training only the head...")
    learn.fit_one_cycle(1, lr/10)
    
    # Use mixed precision if on GPU
    if torch.cuda.is_available():
        learn.to_fp16()
        print("Using mixed precision training")
    
    # Fine-tune with smaller number of epochs
    print(f"Fine tuning for {epochs-1} more epochs...")
    learn.unfreeze()
    learn.fit_one_cycle(epochs-1, slice(lr/100, lr/10))
    
    # Save the model
    learn.export(MODEL_PATH/MODEL_NAME)
    print(f"Model saved to {MODEL_PATH/MODEL_NAME}")
    
    return learn

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train celebrity recognition model with low resources')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training even if GPU is available')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--img-size', type=int, default=IMAGE_SIZE, help=f'Image size (default: {IMAGE_SIZE})')
    parser.add_argument('--max-celebs', type=int, default=MAX_CELEBRITIES, 
                        help=f'Maximum number of celebrities to include (default: {MAX_CELEBRITIES})')
    return parser.parse_args()

def main():
    """Main function to train a model with limited resources."""
    print("Celebrity Recognition Low-Resource Training")
    print("------------------------------------------")
    
    # Parse arguments
    args = parse_arguments()
    
    # Override global variables with arguments
    global BATCH_SIZE, IMAGE_SIZE, MAX_CELEBRITIES
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.img_size
    MAX_CELEBRITIES = args.max_celebs
    
    # Set device
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Configure memory allocation to reduce fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU for training (will be slower)")
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"- Device: {device}")
    print(f"- Image size: {IMAGE_SIZE}px")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Max celebrities: {MAX_CELEBRITIES}")
    
    # Load and limit data
    identity_df = load_identity_data(MAX_CELEBRITIES)
    
    # Setup data loaders
    dataloaders = setup_data_loaders(identity_df, device)
    
    # Train model
    try:
        learn = train_model(dataloaders)
        print("\nTraining completed successfully!")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n\nERROR: GPU ran out of memory!")
            print("Try one or more of these solutions:")
            print("1. Run with --cpu flag to use CPU instead")
            print("2. Reduce --batch-size (current: {BATCH_SIZE})")
            print("3. Reduce --img-size (current: {IMAGE_SIZE})")
            print("4. Reduce --max-celebs (current: {MAX_CELEBRITIES})")
            return 1
        else:
            raise
    
    return 0

if __name__ == "__main__":
    main() 