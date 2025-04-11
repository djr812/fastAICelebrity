#!/usr/bin/env python
"""
Ultra simple prediction script for celebrity recognition
Uses most basic model loading approach
"""

import os
import sys
from pathlib import Path
import torch
from fastai.vision.all import *

# Set environment variable to avoid warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

def find_model_file():
    """Find any available model file"""
    print("Looking for model files...")
    
    # Places to look
    places_to_check = [
        Path("models"),
        Path("."),
        Path(".."),
    ]
    
    # Extensions to try
    extensions = [".pth"]
    
    # Check all possible locations
    for place in places_to_check:
        if not place.exists():
            continue
            
        for file in place.glob("*.pth"):
            print(f"Found model file: {file}")
            return file
            
    print("No model files found")
    return None

def predict_with_model(model_path, image_path):
    """Make predictions using the most basic approach"""
    
    print(f"Loading model from {model_path}...")
    try:
        # Use the parent directory of the image to create a minimal dataloader
        img_dir = Path(image_path).parent
        
        # Create a basic model with resnet34 (smaller than resnet50)
        data = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(),
            get_y=lambda x: "unknown"  # Use a placeholder label
        )
        
        # Create empty dls just to initialize the model
        dls = data.dataloaders(img_dir)
        
        # Create the learner
        learn = vision_learner(dls, resnet34)
        
        # Extract just the filename without extension
        model_filename = Path(model_path).stem
        print(f"Loading weights using filename: {model_filename}...")
        
        # Change to the directory containing the model file
        original_dir = os.getcwd()
        os.chdir(Path(model_path).parent)
        
        # Load the model using just the filename
        learn.load(model_filename)
        
        # Change back to original directory
        os.chdir(original_dir)
        
        # Now make a prediction
        print(f"Processing image: {image_path}")
        img = PILImage.create(image_path)
        
        # Make prediction
        pred, pred_idx, probs = learn.predict(img)
        conf = float(probs[pred_idx])
        
        print(f"\nPrediction Results:")
        print(f"Predicted: {pred}")
        print(f"Confidence: {conf:.2%}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main function"""
    print("Simple Celebrity Recognition")
    print("===========================")
    
    # Check args
    if len(sys.argv) < 2:
        print("Usage: python simple_predict.py <image_path>")
        print("\nNo image provided. Entering interactive mode...")
        
        # Find a model file first
        model_file = find_model_file()
        if model_file is None:
            print("Error: No model file found. Please train a model first.")
            return
            
        # Interactive mode
        while True:
            img_path = input("\nEnter an image path (or 'q' to quit): ")
            if img_path.lower() == 'q':
                break
                
            if not os.path.exists(img_path):
                print(f"Error: File not found: {img_path}")
                continue
                
            predict_with_model(model_file, img_path)
    else:
        # Use command line argument
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"Error: File not found: {img_path}")
            return
            
        # Find a model file
        model_file = find_model_file()
        if model_file is None:
            print("Error: No model file found. Please train a model first.")
            return
            
        # Make prediction
        predict_with_model(model_file, img_path)

if __name__ == "__main__":
    main() 