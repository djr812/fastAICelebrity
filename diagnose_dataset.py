#!/usr/bin/env python
"""
Diagnose dataset script to understand what celebrities are in the training data
and help fix the mismatch between model predictions and actual identities
"""

import os
import sys
import glob
import pickle
from pathlib import Path
from collections import Counter

# Configuration
DATA_DIRS = [
    "data/celeba",       # Common dataset location
    "input/celeba",
    "celeba",
    "dataset",
    "data"
]

VOCAB_FILE = "models/vocab.pkl"
OUTPUT_MAPPING = "celebrity_mapping.csv"

def find_dataset_directory():
    """Find the dataset directory containing celebrity images"""
    print("Looking for dataset directory...")
    
    for data_dir in DATA_DIRS:
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            print(f"Found potential dataset directory: {data_dir}")
            return Path(data_dir)
    
    # Look for any directories with lots of images
    print("Looking for directories with many images...")
    potential_dirs = []
    
    for root, dirs, files in os.walk('.', topdown=True):
        # Skip common non-dataset directories
        if 'venv' in root or '.git' in root or '__pycache__' in root:
            continue
            
        # Check for image files
        jpg_count = len(glob.glob(os.path.join(root, '*.jpg')))
        jpeg_count = len(glob.glob(os.path.join(root, '*.jpeg')))
        png_count = len(glob.glob(os.path.join(root, '*.png')))
        
        total_images = jpg_count + jpeg_count + png_count
        
        if total_images > 50:  # Arbitrary threshold
            potential_dirs.append((root, total_images))
    
    if potential_dirs:
        # Sort by image count (descending)
        potential_dirs.sort(key=lambda x: x[1], reverse=True)
        print(f"Found {len(potential_dirs)} potential dataset directories:")
        for i, (dir_path, count) in enumerate(potential_dirs[:5], 1):
            print(f"{i}. {dir_path} ({count} images)")
        
        # Return the directory with the most images
        return Path(potential_dirs[0][0])
    
    print("No dataset directory found.")
    return None

def analyze_directory_structure(data_dir):
    """Analyze the structure of the dataset directory"""
    print(f"\nAnalyzing directory structure of {data_dir}...")
    
    # Check for CelebA standard structure
    img_dir = data_dir / "img_align_celeba"
    if img_dir.exists() and img_dir.is_dir():
        print(f"Found standard CelebA structure with img_align_celeba directory")
        return "flat", img_dir
    
    # Check for subdirectories (one per celebrity)
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if subdirs:
        # Check if these might be celebrity directories
        total_images = 0
        for subdir in subdirs:
            images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
            total_images += len(images)
        
        if total_images > 50:
            print(f"Found directory structure with {len(subdirs)} potential celebrity subdirectories")
            return "per_celebrity", data_dir
    
    # Check if the directory itself contains many images (flat structure)
    images = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.png"))
    if len(images) > 50:
        print(f"Found flat directory structure with {len(images)} images")
        return "flat", data_dir
    
    print("Could not determine directory structure")
    return "unknown", data_dir

def extract_celebrities_from_per_celebrity_structure(data_dir):
    """Extract celebrities from a structure where each has their own directory"""
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    celebrities = []
    
    print(f"\nExtracting celebrities from {len(subdirs)} subdirectories...")
    
    for subdir in subdirs:
        # Check if directory contains images
        images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
        if images:
            celebrity_name = subdir.name
            celebrities.append((celebrity_name, len(images)))
    
    # Sort by number of images (descending)
    celebrities.sort(key=lambda x: x[1], reverse=True)
    
    return celebrities

def extract_celebrities_from_flat_structure(img_dir):
    """Extract celebrities from a flat directory structure using filenames"""
    print(f"\nExtracting celebrities from flat directory structure...")
    
    # Get all image files
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
    
    # Try to extract celebrity names from filenames
    celebrity_counts = Counter()
    
    for img_path in images:
        # Extract filename without extension
        filename = img_path.stem
        
        # Try different extraction methods
        
        # Method 1: Split by underscore and take first part
        parts = filename.split('_')
        if len(parts) > 1:
            celebrity = parts[0]
            celebrity_counts[celebrity] += 1
            continue
        
        # Method 2: Look for digits - anything before the first digit might be a name
        name_part = ''.join([c for c in filename if not c.isdigit()]).strip()
        if name_part:
            celebrity_counts[name_part] += 1
            continue
        
        # Method 3: Just use the whole filename
        celebrity_counts[filename] += 1
    
    # Convert to list and sort
    celebrities = [(name, count) for name, count in celebrity_counts.most_common()]
    
    return celebrities

def get_current_vocab():
    """Get the current vocabulary from the vocab file"""
    if os.path.exists(VOCAB_FILE):
        try:
            with open(VOCAB_FILE, 'rb') as f:
                vocab_data = pickle.load(f)
                if isinstance(vocab_data, list):
                    return vocab_data
                elif isinstance(vocab_data, dict):
                    # Convert dict to list
                    return [vocab_data[i] for i in sorted(vocab_data.keys())]
                else:
                    print(f"Unknown vocab format: {type(vocab_data)}")
        except Exception as e:
            print(f"Error loading vocab: {e}")
    
    return []

def create_mapping_file(current_vocab, detected_celebrities):
    """Create a mapping file between detected celebrities and current vocab"""
    print(f"\nCreating mapping file: {OUTPUT_MAPPING}")
    
    with open(OUTPUT_MAPPING, 'w') as f:
        f.write("Index,Current Name,Detected Name,Image Count\n")
        
        # Write mapping entries
        for i, current_name in enumerate(current_vocab):
            detected_name = detected_celebrities[i][0] if i < len(detected_celebrities) else ""
            image_count = detected_celebrities[i][1] if i < len(detected_celebrities) else 0
            f.write(f"{i},{current_name},{detected_name},{image_count}\n")
    
    print(f"Mapping file created with {len(current_vocab)} entries")

def main():
    """Main function"""
    print("CelebA Dataset Diagnosis Tool")
    print("============================")
    
    # Find dataset directory
    data_dir = find_dataset_directory()
    if not data_dir:
        print("Error: Could not find dataset directory")
        sys.exit(1)
    
    # Analyze directory structure
    structure, img_dir = analyze_directory_structure(data_dir)
    
    # Extract celebrities based on structure
    if structure == "per_celebrity":
        celebrities = extract_celebrities_from_per_celebrity_structure(img_dir)
    else:
        celebrities = extract_celebrities_from_flat_structure(img_dir)
    
    # Print detected celebrities
    print(f"\nDetected {len(celebrities)} potential celebrities")
    print("\nTop 20 celebrities by image count:")
    for i, (name, count) in enumerate(celebrities[:20], 1):
        print(f"{i}. {name} ({count} images)")
    
    # Get current vocabulary
    current_vocab = get_current_vocab()
    if current_vocab:
        print(f"\nCurrent vocabulary has {len(current_vocab)} entries")
        print("\nFirst 20 names in current vocabulary:")
        for i, name in enumerate(current_vocab[:20], 1):
            print(f"{i}. {name}")
        
        # Create mapping file
        create_mapping_file(current_vocab, celebrities)
        
        print(f"\nNext steps:")
        print(f"1. Review the {OUTPUT_MAPPING} file")
        print(f"2. Edit it to correct any mappings between model indices and actual celebrity names")
        print(f"3. Run the fix_vocab.py script to update your model's vocabulary")
    else:
        print("\nNo existing vocabulary found. Creating celebrities.txt file...")
        with open("celebrities.txt", "w") as f:
            for name, _ in celebrities:
                f.write(f"{name}\n")
        print("Created celebrities.txt with detected celebrity names")

if __name__ == "__main__":
    main() 