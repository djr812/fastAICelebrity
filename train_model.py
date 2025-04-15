#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from fastai.vision.all import *
import torch
import torchvision
import torch.nn as nn

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
DROPOUT_RATE = 0.4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.4
USE_MIXUP = True
USE_TEST_TIME_AUGMENTATION = True
USE_PROGRESSIVE_RESIZING = True

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

# Enhanced data augmentation
def get_transforms(size):
    return [
        # Basic transforms
        Resize(size),
        RandomResizedCrop(size, min_scale=0.8),
        RandomHorizontalFlip(p=0.5),
        
        # Color transforms
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandomBrightness(0.2),
        RandomContrast(0.2),
        
        # Geometric transforms
        RandomRotation(10),
        RandomPerspective(0.2),
        
        # Normalization
        Normalize.from_stats(*imagenet_stats)
    ]

# Progressive resizing
def get_progressive_sizes():
    return [128, 192, 256, 320]

# Enhanced model creation
def create_model(num_classes, architecture='resnet101', dropout_rate=0.4):
    model = getattr(torchvision.models, architecture)(pretrained=True)
    
    # Replace final layer with custom head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(dropout_rate),
        nn.Linear(1024, num_classes)
    )
    
    return model

# Enhanced training loop
def train_model(dls, num_classes, model_path, num_epochs=20, fine_tune_epochs=15):
    # Create model with enhanced architecture
    model = create_model(num_classes, ARCHITECTURE, DROPOUT_RATE)
    
    # Create learner with enhanced callbacks
    learn = Learner(
        dls,
        model,
        loss_func=LabelSmoothingCrossEntropy() if LABEL_SMOOTHING > 0 else CrossEntropyLoss(),
        metrics=[accuracy, top_k_accuracy],
        cbs=[
            MixedPrecision(),
            GradientAccumulation(GRAD_ACCUM),
            EarlyStopping(patience=5),
            SaveModelCallback(monitor='valid_loss'),
            ReduceLROnPlateau(monitor='valid_loss', patience=3),
            MixUp(alpha=MIXUP_ALPHA) if USE_MIXUP else None,
            TestTimeAugmentation() if USE_TEST_TIME_AUGMENTATION else None
        ]
    )
    
    # Progressive resizing training
    if USE_PROGRESSIVE_RESIZING:
        for size in get_progressive_sizes():
            dls = get_dls(size)
            learn.dls = dls
            learn.fit_one_cycle(num_epochs, LEARNING_RATE)
    else:
        learn.fit_one_cycle(num_epochs, LEARNING_RATE)
    
    # Fine-tuning
    learn.unfreeze()
    learn.fit_one_cycle(fine_tune_epochs, LEARNING_RATE/10)
    
    # Save final model
    learn.save(model_path)
    
    return learn

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
    train_model(dataloaders, args.arch, args.epochs)

if __name__ == "__main__":
    main() 