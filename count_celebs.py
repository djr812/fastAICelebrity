#!/usr/bin/env python
import pandas as pd
from collections import Counter
from pathlib import Path

# Configuration
DATA_DIR = Path("/home/dave/torch/celeba_images/img_align_celeba")
IDENTITY_FILE = "list_identity_celeba.txt"

def load_identity_data():
    """Load celebrity identity data from the text file."""
    print("Loading identity data...")
    
    # Skip the first few lines which contain metadata
    df = pd.read_csv(IDENTITY_FILE, sep="\s+", skiprows=1, 
                     names=["image_id", "identity_name"])
    print(f"Loaded {len(df)} image identity records")
    return df

def analyze_celeb_counts(df, min_images=20):
    """Analyze how many celebrities have at least min_images."""
    # Count images per celebrity
    celeb_counts = Counter(df['identity_name'])
    
    # Filter celebrities with enough images
    eligible_celebs = [celeb for celeb, count in celeb_counts.items() 
                      if count >= min_images]
    
    # Print statistics
    print(f"\nTotal celebrities in dataset: {len(celeb_counts)}")
    print(f"Celebrities with at least {min_images} images: {len(eligible_celebs)}")
    
    # Print top 10 celebrities by image count
    print("\nTop 10 celebrities by image count:")
    for celeb, count in sorted(celeb_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{celeb}: {count} images")
    
    return len(eligible_celebs)

def main():
    # Load identity data
    identity_df = load_identity_data()
    
    # Analyze counts
    eligible_count = analyze_celeb_counts(identity_df)
    
    print(f"\nYou can train on up to {eligible_count} celebrities with at least 20 images each")

if __name__ == "__main__":
    main() 