#!/usr/bin/env python
"""
Simplified prediction script that uses manual overrides to correct predictions.
This script works with your existing model but overrides incorrect predictions.
"""

import os
import sys
import pickle
from pathlib import Path
from fastai.vision.all import *

# Configuration
OVERRIDE_FILE = "manual_overrides.csv"
MODEL_PATH = Path("models/celebrity_recognition_model.pth")
VOCAB_FILE = Path("models/vocab.pkl")

def load_manual_overrides():
    """Load manual overrides from CSV file"""
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
            
            if overrides:
                print(f"Loaded {len(overrides)} manual overrides")
        except Exception as e:
            print(f"Error loading manual overrides: {e}")
    
    return overrides

def find_best_model():
    """Find best available model file"""
    if MODEL_PATH.exists():
        return MODEL_PATH
    
    # Look for other models
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        if model_files:
            # Sort by file size (descending)
            model_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
            return model_files[0]
    
    # Look in current directory
    model_files = list(Path(".").glob("*.pth"))
    if model_files:
        return model_files[0]
    
    return None

def load_vocab():
    """Load vocabulary mapping"""
    if VOCAB_FILE.exists():
        try:
            with open(VOCAB_FILE, 'rb') as f:
                vocab_data = pickle.load(f)
                if isinstance(vocab_data, list):
                    print(f"Loaded {len(vocab_data)} names from vocabulary")
                    return vocab_data
                elif isinstance(vocab_data, dict):
                    # Convert to list
                    names = [vocab_data[i] for i in sorted(vocab_data.keys())]
                    print(f"Loaded {len(names)} names from vocabulary dictionary")
                    return names
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
    
    # Try celebrities.txt
    if os.path.exists("celebrities.txt"):
        with open("celebrities.txt", 'r') as f:
            names = [line.strip() for line in f if line.strip()]
            if names:
                print(f"Loaded {len(names)} names from celebrities.txt")
                return names
    
    # Return generic names
    print("Using generic celebrity names")
    return [f"Celebrity_{i}" for i in range(100)]

def predict_with_override(image_path, model_path=None):
    """Predict with manual override"""
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return False
        
        # Find model if not specified
        if model_path is None:
            model_path = find_best_model()
            if model_path is None:
                print("Error: No model file found")
                return False
            print(f"Using model: {model_path}")
        
        # Load model using FastAI
        try:
            print(f"Loading model from {model_path}")
            
            # Try different loading approaches
            try:
                # Try direct loading first
                learn = load_learner(model_path)
                print("Model loaded successfully!")
            except Exception as e:
                # Create minimal learner with ResNet architecture
                print(f"Direct loading failed: {e}")
                print("Creating minimal learner...")
                
                # Create simple data block
                dls = DataBlock(
                    blocks=(ImageBlock, CategoryBlock),
                    get_items=get_image_files,
                    get_y=lambda x: Path(x).parent.name,
                    item_tfms=Resize(224)
                ).dataloaders(Path("."))
                
                # Create learner
                learn = vision_learner(dls, resnet50, pretrained=False)
                
                # Load weights with strict=False
                state = torch.load(model_path, map_location='cpu')
                
                # Try to extract state dict
                if isinstance(state, dict):
                    if 'model' in state:
                        state_dict = state['model']
                    elif 'state_dict' in state:
                        state_dict = state['state_dict']
                    elif 'model_state_dict' in state:
                        state_dict = state['model_state_dict']
                    else:
                        state_dict = state
                else:
                    state_dict = state
                
                # Load state dict
                learn.model.load_state_dict(state_dict, strict=False)
                print("Model loaded with partial weights")
            
            # Load vocabulary
            vocab = load_vocab()
            
            # Load overrides
            overrides = load_manual_overrides()
            
            # Process image
            print(f"Processing image: {image_path}")
            img = PILImage.create(image_path)
            
            # Make prediction
            print("Making prediction...")
            pred_class, pred_idx, probs = learn.predict(img)
            
            # Get top 5 predictions
            top_idxs = probs.argsort(descending=True)[:5]
            top_probs = probs[top_idxs]
            
            # Display results
            print("\n==== Prediction Results ====")
            
            # Get top prediction
            pred_idx_val = int(top_idxs[0])
            pred_name = vocab[pred_idx_val] if pred_idx_val < len(vocab) else f"Unknown_{pred_idx_val}"
            
            # Apply manual override
            if pred_name in overrides:
                corrected_name = overrides[pred_name]
                print(f"MANUAL OVERRIDE: {pred_name} -> {corrected_name}")
                pred_name = corrected_name
            
            # Display top prediction
            print(f"\nPredicted Celebrity: {pred_name}")
            print(f"Confidence: {top_probs[0]:.4f} ({top_probs[0]*100:.1f}%)")
            
            # Display top 5
            print("\nTop 5 Matches:")
            print("-" * 40)
            for i, (idx, prob) in enumerate(zip(top_idxs, top_probs)):
                idx_val = int(idx)
                name = vocab[idx_val] if idx_val < len(vocab) else f"Unknown_{idx_val}"
                
                # Apply override if needed
                if name in overrides:
                    corrected = overrides[name]
                    override_mark = f" -> {corrected}"
                else:
                    override_mark = ""
                
                print(f"{i+1}. {name:<25} {prob:.4f} ({prob*100:.1f}%){override_mark}")
            
            # Extract name from filename for comparison
            filename = Path(image_path).stem
            name_from_file = filename.replace('_', ' ')
            print(f"\nFilename: {name_from_file}")
            
            return True
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main function"""
    print("Celebrity Recognition with Manual Overrides")
    print("=========================================")
    
    # Check if override file exists
    if not os.path.exists(OVERRIDE_FILE):
        print(f"Warning: Override file {OVERRIDE_FILE} not found")
        print("You can create it using edit_overrides.py")
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python predict_with_override.py <image_path>")
        
        # Look for test images
        test_dirs = ["testpics", "sample_images", "."]
        for dir_path in test_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                import glob
                images = []
                for ext in [".jpg", ".jpeg", ".png"]:
                    images.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
                if images:
                    print(f"\nExample: python predict_with_override.py {images[0]}")
                    break
        
        return
    
    # Get image path
    image_path = sys.argv[1]
    
    # Run prediction
    predict_with_override(image_path)

if __name__ == "__main__":
    main() 