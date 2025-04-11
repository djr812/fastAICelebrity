#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from fastai.vision.all import *

# Configuration
PROCESSED_DIR = Path("processed_celeba")
MODEL_PATH = Path("models")
MODEL_NAME = "celebrity_recognition_model.pkl"
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 2e-3
ARCHITECTURE = "resnet50"

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
                           max_lighting=0.2, max_warp=0.2, p_affine=0.75, 
                           p_lighting=0.75),
            Normalize.from_stats(*imagenet_stats)
        ]
    )
    
    # Create dataloaders from the processed directory
    dls = celeb_data.dataloaders(data_path, bs=batch_size)
    print(f"Created dataloaders with {len(dls.train_ds)} training and {len(dls.valid_ds)} validation images")
    print(f"Number of classes: {len(dls.vocab)}")
    
    # Show a batch to verify
    print("Showing a sample batch:")
    dls.show_batch(max_n=4, nrows=1)
    
    return dls

def train_model(dls, architecture=ARCHITECTURE, epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """Train the celebrity recognition model."""
    print("\nTraining model...")
    
    # Create a CNN learner using the selected architecture
    arch = getattr(models.resnet, architecture)
    learn = vision_learner(dls, arch, metrics=[error_rate, accuracy])
    
    # Find optimal learning rate
    lr_finder = learn.lr_find()
    suggested_lr = lr_finder.valley
    print(f"Suggested learning rate: {suggested_lr}")
    
    # Train the model using 1cycle policy
    print(f"Training for {epochs} epochs with learning rate {lr}")
    learn.fine_tune(epochs, lr)
    
    # Evaluate model
    print("\nEvaluating model...")
    valid_loss, valid_accuracy = learn.validate()
    print(f"Validation loss: {valid_loss:.4f}")
    print(f"Validation accuracy: {valid_accuracy:.4f}")
    
    # Show confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
    
    # Show most confused classes
    print("\nMost confused classes:")
    interp.plot_top_losses(9, figsize=(15, 11))
    
    # Save the model
    os.makedirs(MODEL_PATH, exist_ok=True)
    learn.export(MODEL_PATH/MODEL_NAME)
    print(f"Model saved to {MODEL_PATH/MODEL_NAME}")
    
    return learn

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train celebrity recognition model')
    parser.add_argument('--data-dir', type=str, default=str(PROCESSED_DIR),
                       help=f'Path to processed dataset directory (default: {PROCESSED_DIR})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--img-size', type=int, default=IMAGE_SIZE,
                       help=f'Image size for training (default: {IMAGE_SIZE})')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help=f'Number of epochs to train (default: {NUM_EPOCHS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--arch', type=str, default=ARCHITECTURE,
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                       help=f'Model architecture (default: {ARCHITECTURE})')
    parser.add_argument('--model-path', type=str, default=str(MODEL_PATH),
                       help=f'Path to save model (default: {MODEL_PATH})')
    return parser.parse_args()

def main():
    """Main function to train the recognition model."""
    print("Celebrity Recognition Model Training")
    print("===================================")
    
    # Parse arguments
    args = parse_arguments()
    
    # Set global variables from arguments
    global PROCESSED_DIR, MODEL_PATH, MODEL_NAME
    PROCESSED_DIR = Path(args.data_dir)
    MODEL_PATH = Path(args.model_path)
    
    # Check if processed dataset exists
    if not PROCESSED_DIR.exists():
        print(f"Error: Processed dataset not found at {PROCESSED_DIR}")
        print("Please run preprocess_dataset.py first")
        return 1
    
    # Set up data loaders
    dataloaders = setup_data_loaders(PROCESSED_DIR, args.img_size, args.batch_size)
    
    # Train model
    learn = train_model(dataloaders, args.arch, args.epochs, args.lr)
    
    print("\nTraining complete!")
    print(f"You can now use the model for prediction with predict.py")
    
    return 0

if __name__ == "__main__":
    main() 