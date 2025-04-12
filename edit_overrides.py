#!/usr/bin/env python
"""
Simple script to create manual overrides for specific images.
This is the fastest way to correct predictions for specific images.
"""

import os
import csv
import glob
from pathlib import Path

# Configuration
OVERRIDE_FILE = "manual_overrides.csv"
TEST_DIRS = ["testpics", "sample_images", "."]

def load_existing_overrides():
    """Load existing manual overrides"""
    overrides = {}
    
    if os.path.exists(OVERRIDE_FILE):
        try:
            with open(OVERRIDE_FILE, 'r') as f:
                # Skip header
                f.readline()
                # Read overrides
                for line in f:
                    if ',' in line:
                        predicted, correct = line.strip().split(',', 1)
                        overrides[predicted] = correct
            
            print(f"Loaded {len(overrides)} existing manual overrides")
        except Exception as e:
            print(f"Error loading manual overrides: {e}")
    
    return overrides

def save_overrides(overrides):
    """Save overrides to CSV file"""
    try:
        with open(OVERRIDE_FILE, 'w') as f:
            f.write("predicted,correct\n")
            for predicted, correct in overrides.items():
                f.write(f"{predicted},{correct}\n")
        
        print(f"Saved {len(overrides)} overrides to {OVERRIDE_FILE}")
        return True
    except Exception as e:
        print(f"Error saving overrides: {e}")
        return False

def find_images():
    """Find test images to use for creating overrides"""
    images = []
    
    for dir_path in TEST_DIRS:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # Look for image files
            for ext in [".jpg", ".jpeg", ".png"]:
                pattern = os.path.join(dir_path, f"*{ext}")
                found = glob.glob(pattern)
                images.extend(found)
    
    return sorted(images)

def extract_name_from_filename(filename):
    """Extract potential celebrity name from filename"""
    name = Path(filename).stem
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    
    return name

def create_overrides_from_images():
    """Create overrides based on image filenames"""
    # Find images
    images = find_images()
    if not images:
        print("No images found")
        return
    
    print(f"Found {len(images)} images")
    
    # Load existing overrides
    overrides = load_existing_overrides()
    
    # Process each image
    for i, img_path in enumerate(images, 1):
        # Extract potential name from filename
        name = extract_name_from_filename(img_path)
        
        print(f"\nImage {i}/{len(images)}: {img_path}")
        print(f"Extracted name: {name}")
        
        # Ask for override
        predicted = input("Enter predicted name (or press Enter to skip): ").strip()
        if not predicted:
            continue
        
        correct = input(f"Enter correct name [default: {name}]: ").strip()
        if not correct:
            correct = name
        
        # Add override
        overrides[predicted] = correct
        print(f"Added override: {predicted} -> {correct}")
    
    # Save overrides
    if overrides:
        save_overrides(overrides)
        print("\nOverrides created successfully!")
        print("Your prediction scripts will now use these overrides.")
    else:
        print("\nNo overrides created.")

def create_override_from_input():
    """Create a single override from user input"""
    # Load existing overrides
    overrides = load_existing_overrides()
    
    # Get input from user
    predicted = input("Enter predicted name (e.g., 'Fonzworth_Bentley'): ").strip()
    if not predicted:
        print("Cancelled")
        return
    
    correct = input("Enter correct name (e.g., 'Zuleikha_Robinson'): ").strip()
    if not correct:
        print("Cancelled")
        return
    
    # Add override
    overrides[predicted] = correct
    print(f"Added override: {predicted} -> {correct}")
    
    # Save overrides
    save_overrides(overrides)
    print("\nOverride added successfully!")
    print("Your prediction scripts will now use this override.")

def main():
    """Main function"""
    print("Manual Override Editor")
    print("=====================")
    
    print("\nThis tool lets you create manual overrides for celebrity predictions.")
    print("There are two ways to create overrides:")
    print("1. Process image files and extract names from filenames")
    print("2. Directly enter a predicted name and its correction")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        create_overrides_from_images()
    elif choice == "2":
        create_override_from_input()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 