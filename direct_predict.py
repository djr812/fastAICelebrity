#!/usr/bin/env python
"""
Ultra-simple direct prediction script for celebrity recognition.
This script avoids architecture creation and loads directly from the saved model.
"""

import os
import sys
import pickle
from pathlib import Path
from fastai.vision.all import *

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
MODEL_DIR = Path("models")
DEFAULT_MODEL = "celebrity_recognition_model.pth"
VOCAB_FILE = MODEL_DIR/"vocab.pkl"

def load_celebrity_names():
    """Load celebrity names from various possible sources"""
    # First try vocab.pkl
    if VOCAB_FILE.exists():
        try:
            print(f"Loading vocabulary from {VOCAB_FILE}")
            with open(VOCAB_FILE, 'rb') as f:
                vocab_data = pickle.load(f)
                if isinstance(vocab_data, list):
                    print(f"Found {len(vocab_data)} celebrities in list format")
                    return vocab_data
                elif isinstance(vocab_data, dict):
                    print(f"Found {len(vocab_data)} celebrities in dict format")
                    # Sort the dict by index for consistent results
                    return [vocab_data[i] for i in sorted(vocab_data.keys())]
                else:
                    print(f"Unknown vocab format: {type(vocab_data)}")
        except Exception as e:
            print(f"Error loading vocab: {e}")
    
    # Try celebrities.txt as fallback
    if os.path.exists("celebrities.txt"):
        try:
            with open("celebrities.txt", "r") as f:
                names = [line.strip() for line in f if line.strip()]
                if names:
                    print(f"Loaded {len(names)} names from celebrities.txt")
                    return names
        except Exception as e:
            print(f"Error loading from celebrities.txt: {e}")
    
    # Default to generic names
    print("Using generic celebrity names")
    return [f"Celebrity_{i}" for i in range(100)]

def load_learner_direct(model_path):
    """Load learner with direct approach, avoiding architecture creation"""
    print(f"Loading model from {model_path}")
    
    # Get model directory (for relative paths)
    model_dir = Path(model_path).parent
    
    # Try multiple approaches to load the model
    try:
        # First approach: direct load with full path
        try:
            print("Attempting direct learner load...")
            learn = load_learner(model_path)
            print("Model loaded successfully!")
            return learn
        except Exception as e:
            print(f"Direct load failed: {e}")
        
        # Second approach: change to directory and load with basename
        try:
            print("Trying directory-based loading...")
            # Store current directory
            orig_dir = os.getcwd()
            # Change to model directory
            os.chdir(model_dir)
            # Load with just the filename
            learn = load_learner(Path(model_path).name)
            # Return to original directory
            os.chdir(orig_dir)
            print("Model loaded successfully!")
            return learn
        except Exception as e:
            print(f"Directory-based loading failed: {e}")
            # Make sure we return to original directory
            if orig_dir:
                os.chdir(orig_dir)
        
        # Third approach: create a minimal learner and load state dict
        print("Creating minimal learner...")
        # Create a simple dataloader
        dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            get_y=lambda x: Path(x).parent.name,
            item_tfms=Resize(224),
        ).dataloaders(Path("."))
        
        # Create learner without pretrained weights
        learn = vision_learner(dls, resnet50, pretrained=False)
        
        # Try to load state dict directly
        try:
            print("Loading state dict directly...")
            state = torch.load(model_path, map_location='cpu')
            # If state contains 'model', use that
            if isinstance(state, dict) and 'model' in state:
                print("Found 'model' key in state dict")
                learn.model.load_state_dict(state['model'], strict=False)
            # If it's a plain state dict, try loading it
            else:
                print("Loading plain state dict")
                learn.model.load_state_dict(state, strict=False)
            print("Model loaded with partial weights!")
            return learn
        except Exception as e:
            print(f"State dict loading failed: {e}")
            raise ValueError("All loading methods failed")
    
    except Exception as e:
        print(f"All loading attempts failed: {e}")
        sys.exit(1)

def predict_celebrity(model_path, image_path):
    """Main prediction function with robust error handling"""
    try:
        # First, check files exist
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print(f"Current directory: {os.getcwd()}")
            sys.exit(1)
            
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            sys.exit(1)
        
        # Load celebrity names
        celebrity_names = load_celebrity_names()
        
        # Load the model
        learn = load_learner_direct(model_path)
        
        # Process the image
        print(f"Processing image: {image_path}")
        img = PILImage.create(image_path)
        
        # Make prediction - with error handling since the model might be partial
        try:
            print("Making prediction...")
            pred_class, pred_idx, probs = learn.predict(img)
            
            # Display top predictions
            print("\n==== Prediction Results ====")
            
            # Get top 5 predictions
            top_idxs = probs.argsort(descending=True)[:5]
            top_probs = probs[top_idxs]
            
            for i, (idx, prob) in enumerate(zip(top_idxs, top_probs)):
                idx_val = int(idx)
                name = celebrity_names[idx_val] if idx_val < len(celebrity_names) else f"Unknown_{idx_val}"
                print(f"{i+1}. {name}: {prob:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            
            # Show model info for debugging
            print("\nModel information:")
            print(f"Model type: {type(learn.model)}")
            return False
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    print("Direct Celebrity Recognition")
    print("==========================")
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python direct_predict.py <image_path> [model_path]")
        sys.exit(1)
    
    # Get image path
    image_path = sys.argv[1]
    
    # Get model path (default or provided)
    model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR/DEFAULT_MODEL
    
    # Make prediction
    predict_celebrity(model_path, image_path)

if __name__ == "__main__":
    main() 