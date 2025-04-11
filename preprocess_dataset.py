#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from PIL import Image
import argparse
from collections import Counter
import random

# Configuration
DATA_DIR = Path("img_align_celeba")
IDENTITY_FILE = "list_identity_celeba.txt"
PROCESSED_DIR = Path("processed_celeba")
MIN_IMAGES_PER_CELEB = 20  # Minimum number of images per celebrity
MAX_CELEBS = 100  # Maximum number of celebrities to include (None for all)

def load_identity_data():
    """Load celebrity identity data from the text file."""
    print("Loading identity data...")
    
    # Skip the first few lines which contain metadata
    df = pd.read_csv(IDENTITY_FILE, sep="\s+", skiprows=1, 
                     names=["image_id", "identity_name"])
    print(f"Loaded {len(df)} image identity records")
    return df

def preprocess_dataset(df, min_images=MIN_IMAGES_PER_CELEB, max_celebs=MAX_CELEBS):
    """
    Preprocess the dataset:
    1. Filter out celebrities with too few images
    2. Limit to a maximum number of celebrities
    3. Create a balanced dataset structure
    """
    print("\nPreprocessing dataset...")
    
    # Count images per celebrity
    celeb_counts = Counter(df['identity_name'])
    print(f"Found {len(celeb_counts)} unique celebrities")
    
    # Filter celebrities with enough images
    eligible_celebs = [celeb for celeb, count in celeb_counts.items() 
                      if count >= min_images]
    print(f"Found {len(eligible_celebs)} celebrities with at least {min_images} images")
    
    # Limit number of celebrities if specified
    if max_celebs and len(eligible_celebs) > max_celebs:
        print(f"Limiting to top {max_celebs} celebrities by image count")
        celeb_counts_filtered = {celeb: count for celeb, count in celeb_counts.items() 
                                if celeb in eligible_celebs}
        eligible_celebs = [celeb for celeb, _ in sorted(celeb_counts_filtered.items(), 
                                                      key=lambda x: x[1], reverse=True)[:max_celebs]]
    
    # Filter dataset to only include selected celebrities
    filtered_df = df[df['identity_name'].isin(eligible_celebs)].copy()
    print(f"Filtered dataset contains {len(filtered_df)} images of {len(eligible_celebs)} celebrities")
    
    return filtered_df, eligible_celebs

def organize_dataset(df, eligible_celebs):
    """
    Organize dataset into a directory structure suitable for fastai:
    processed_celeba/
    ├── train/
    │   ├── celebrity1/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   ├── celebrity2/
    │   ...
    ├── valid/
    │   ├── celebrity1/
    │   ...
    """
    print("\nOrganizing dataset into train/validation splits...")
    
    # Create directory structure
    train_dir = PROCESSED_DIR / "train"
    valid_dir = PROCESSED_DIR / "valid"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    
    # Create directories for each celebrity
    for celeb in eligible_celebs:
        os.makedirs(train_dir / celeb, exist_ok=True)
        os.makedirs(valid_dir / celeb, exist_ok=True)
    
    # Organize files
    validation_ratio = 0.2  # 20% for validation
    files_copied = 0
    
    for celeb in eligible_celebs:
        # Get all images for this celebrity
        celeb_images = df[df['identity_name'] == celeb]['image_id'].values
        
        # Shuffle and split
        random.shuffle(celeb_images)
        split_idx = int(len(celeb_images) * (1 - validation_ratio))
        
        train_images = celeb_images[:split_idx]
        valid_images = celeb_images[split_idx:]
        
        # Copy training images
        for img_id in train_images:
            src_path = DATA_DIR / img_id
            dst_path = train_dir / celeb / img_id
            if src_path.exists():
                shutil.copy(src_path, dst_path)
                files_copied += 1
        
        # Copy validation images
        for img_id in valid_images:
            src_path = DATA_DIR / img_id
            dst_path = valid_dir / celeb / img_id
            if src_path.exists():
                shutil.copy(src_path, dst_path)
                files_copied += 1
        
        print(f"Processed {celeb}: {len(train_images)} training, {len(valid_images)} validation images")
    
    print(f"Total files copied: {files_copied}")
    
    # Generate statistics
    train_counts = {celeb: len(list((train_dir / celeb).glob('*.jpg'))) for celeb in eligible_celebs}
    valid_counts = {celeb: len(list((valid_dir / celeb).glob('*.jpg'))) for celeb in eligible_celebs}
    
    total_train = sum(train_counts.values())
    total_valid = sum(valid_counts.values())
    
    print(f"\nDataset organization complete:")
    print(f"Training images: {total_train}")
    print(f"Validation images: {total_valid}")
    print(f"Total images: {total_train + total_valid}")
    
    # Save celebrity list for reference
    with open(PROCESSED_DIR / "celebrity_list.txt", "w") as f:
        for celeb in eligible_celebs:
            f.write(f"{celeb}\n")
    
    print(f"Celebrity list saved to {PROCESSED_DIR / 'celebrity_list.txt'}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess CelebA dataset for facial recognition')
    parser.add_argument('--min-images', type=int, default=MIN_IMAGES_PER_CELEB,
                       help=f'Minimum images per celebrity (default: {MIN_IMAGES_PER_CELEB})')
    parser.add_argument('--max-celebs', type=int, default=MAX_CELEBS,
                       help=f'Maximum number of celebrities to include (default: {MAX_CELEBS})')
    parser.add_argument('--output-dir', type=str, default=str(PROCESSED_DIR),
                       help=f'Output directory for processed dataset (default: {PROCESSED_DIR})')
    return parser.parse_args()

def main():
    """Main function to preprocess the dataset."""
    print("CelebA Dataset Preprocessing")
    print("===========================")
    
    # Parse arguments
    args = parse_arguments()
    global PROCESSED_DIR
    PROCESSED_DIR = Path(args.output_dir)
    
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Load identity data
    identity_df = load_identity_data()
    
    # Preprocess dataset
    filtered_df, eligible_celebs = preprocess_dataset(identity_df, args.min_images, args.max_celebs)
    
    # Organize dataset
    organize_dataset(filtered_df, eligible_celebs)
    
    print("\nPreprocessing complete!")
    print(f"Preprocessed dataset is available at: {PROCESSED_DIR}")

if __name__ == "__main__":
    main() 