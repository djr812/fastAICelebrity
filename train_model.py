#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from fastai.vision.all import *
import torch

# Configuration
PROCESSED_DIR = Path("processed_celeba")
MODEL_PATH = Path("models")
MODEL_NAME = "celebrity_recognition_model.pkl"
IMAGE_SIZE = 224
BATCH_SIZE = 16  # Reduced from 64
NUM_EPOCHS = 10
LEARNING_RATE = 2e-3
ARCHITECTURE = "resnet50"
GRAD_ACCUM = 4  # Gradient accumulation steps

def setup_data_loaders(data_path, img_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """Set up FastAI data loaders for training."""
    print("Setting up data loaders...")
    
    # Create FastAI DataBlock for handling image data
    celeb_data = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='train', valid_name='valid'),
        get_y=parent_label,
        item_tfms=Resize(img_size),
        batch_tfms=[
            *aug_transforms(do_flip=True, flip_vert=False, max_rotate=10.0, 
                           max_lighting=0.2, max_warp=0.2, p_affine=0.75),
            Normalize.from_stats(*imagenet_stats)
        ]
    )
    
    # Create dataloaders
    dls = celeb_data.dataloaders(data_path, bs=batch_size)
    return dls

def train_model(dls, architecture=ARCHITECTURE, epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """Train the celebrity recognition model."""
    print("Initializing model...")
    
    # Create learner with mixed precision
    learn = vision_learner(dls, architecture, metrics=accuracy, 
                          cbs=[MixedPrecision(), GradientAccumulation(GRAD_ACCUM)])
    
    # Find optimal learning rate
    print("Finding optimal learning rate...")
    suggested_lr = learn.lr_find()
    print(f"Suggested learning rate: {suggested_lr}")
    
    # Train the model
    print(f"Training model for {epochs} epochs...")
    learn.fit_one_cycle(epochs, suggested_lr)
    
    # Save the model
    os.makedirs(MODEL_PATH, exist_ok=True)
    learn.export(MODEL_PATH / MODEL_NAME)
    print(f"Model saved to {MODEL_PATH / MODEL_NAME}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train celebrity recognition model')
    parser.add_argument('--img_size', type=int, default=IMAGE_SIZE,
                        help='Size of input images')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--arch', type=str, default=ARCHITECTURE,
                        help='Model architecture')
    return parser.parse_args()

def main():
    """Main function to train the model."""
    args = parse_arguments()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will be very slow on CPU.")
    
    # Set up data loaders
    dataloaders = setup_data_loaders(PROCESSED_DIR, args.img_size, args.batch_size)
    
    # Train the model
    train_model(dataloaders, args.arch, args.epochs, args.lr)

if __name__ == "__main__":
    main() 