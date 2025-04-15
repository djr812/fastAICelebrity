#!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import shutil

# Configuration
DATA_DIR = Path("/home/dave/torch/celeba_images/img_align_celeba")
IDENTITY_FILE = "list_identity_celeba.txt"
SAMPLE_DIR = Path("sample_images")
SAMPLE_COUNT = 5

def load_identity_data():
    """Load celebrity identity data from the text file."""
    print("Loading identity data...")
    # Skip the first few lines which contain metadata
    df = pd.read_csv(IDENTITY_FILE, sep=r"\s+", skiprows=1, 
                     names=["image_id", "identity_name"])
    print(f"Loaded {len(df)} image identity records")
    return df

def analyze_dataset(df):
    """Analyze the dataset and print statistics."""
    print("\nDataset Statistics:")
    print(f"Total images: {len(df)}")
    
    # Count unique celebrities
    celebrities = df['identity_name'].unique()
    print(f"Unique celebrities: {len(celebrities)}")
    
    # Images per celebrity (top 10)
    celebs_count = df['identity_name'].value_counts()
    print("\nTop 10 celebrities by image count:")
    print(celebs_count.head(10))
    
    # Plot distribution of top 20 celebrities
    plt.figure(figsize=(12, 6))
    celebs_count.head(20).plot(kind='bar')
    plt.title('Number of Images per Celebrity (Top 20)')
    plt.xlabel('Celebrity')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig('celebrity_distribution.png')
    print("Distribution plot saved as 'celebrity_distribution.png'")

def display_sample_images(df):
    """Display and save sample images from the dataset."""
    # Create sample directory
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    # Get random sample of celebrities with at least 5 images
    celebs_with_samples = df['identity_name'].value_counts()[df['identity_name'].value_counts() >= SAMPLE_COUNT]
    sample_celebs = random.sample(list(celebs_with_samples.index), min(5, len(celebs_with_samples)))
    
    print(f"\nSaving sample images for {len(sample_celebs)} celebrities...")
    
    for celeb in sample_celebs:
        celeb_images = df[df['identity_name'] == celeb]['image_id'].values
        sample_images = random.sample(list(celeb_images), min(SAMPLE_COUNT, len(celeb_images)))
        
        # Create directory for this celebrity
        celeb_dir = SAMPLE_DIR / celeb
        os.makedirs(celeb_dir, exist_ok=True)
        
        # Copy images to sample directory
        for img_id in sample_images:
            src_path = DATA_DIR / img_id
            dst_path = celeb_dir / img_id
            shutil.copy(src_path, dst_path)
        
        print(f"  - Saved {len(sample_images)} images for {celeb}")
    
    print(f"Sample images saved to {SAMPLE_DIR}")

def main():
    """Main function to analyze and prepare dataset."""
    print("Celebrity Dataset Analysis and Preparation")
    print("------------------------------------------")
    
    # Load identity data
    identity_df = load_identity_data()
    
    # Analyze dataset
    analyze_dataset(identity_df)
    
    # Save sample images
    display_sample_images(identity_df)

if __name__ == "__main__":
    main() 