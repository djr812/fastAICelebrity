#!/usr/bin/env python
"""
Simple FastAI-based prediction script for celebrity recognition.
This uses the FastAI libraries that were used for training.
"""

import os
import sys
from pathlib import Path
from fastai.vision.all import *

# Configuration
MODEL_PATH = Path("models/celebrity_recognition_model.pth")
CELEBRITIES_FILE = Path("celebrities.txt")

def load_celebrity_names():
    """Load celebrity names from celebrities.txt file"""
    if CELEBRITIES_FILE.exists():
        with open(CELEBRITIES_FILE, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: {CELEBRITIES_FILE} not found")
        return [f"Celebrity_{i}" for i in range(100)]

def predict_image(model_path, image_path):
    """Predict celebrity from image using FastAI learner"""
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return False
            
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return False
            
        print(f"Loading model from {model_path}...")
        
        # Get list of celebrity names
        celebrity_names = load_celebrity_names()
        
        # Create a DataBlock with the same classes as your training
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            get_y=lambda x: Path(x).parent.name,  # Placeholder
            splitter=RandomSplitter()
        )
        
        # Create a minimal dataloader to support the model
        temp_path = Path(image_path).parent
        dls = dblock.dataloaders(temp_path)
        
        # Create learner with appropriate architecture
        learner = vision_learner(dls, resnet50, pretrained=False)
        
        # Load the model weights
        print("Loading model weights...")
        try:
            # Get the model filename without extension
            model_file = Path(model_path).stem
            learner.load(model_file)
        except Exception as e:
            # Try loading with full path
            print(f"Standard loading failed: {e}")
            print("Trying alternative loading method...")
            state = torch.load(model_path, map_location='cpu')
            learner.model.load_state_dict(state)
        
        # Load the image
        print(f"Processing image: {image_path}")
        img = PILImage.create(image_path)
        
        # Get prediction
        print("Making prediction...")
        pred_class, pred_idx, probs = learner.predict(img)
        
        # Get top 5 predictions
        top_idxs = probs.argsort(descending=True)[:5]
        top_probs = probs[top_idxs]
        
        # Display results
        print("\n==== Prediction Results ====")
        for i, (idx, prob) in enumerate(zip(top_idxs, top_probs)):
            idx_int = int(idx)
            name = celebrity_names[idx_int] if idx_int < len(celebrity_names) else f"Unknown_{idx_int}"
            print(f"{i+1}. {name}: {prob:.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Celebrity Recognition with FastAI")
    print("================================")
    
    # Get image path from command line
    if len(sys.argv) < 2:
        print("Usage: python fastai_predict.py <image_path> [model_path]")
        print("\nExample: python fastai_predict.py testpics/celebrity.jpg")
        sys.exit(1)
    
    # Get paths
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
    
    # Run prediction
    predict_image(model_path, image_path)

if __name__ == "__main__":
    main() 