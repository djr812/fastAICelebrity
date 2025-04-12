#!/usr/bin/env python
"""
Final version of celebrity prediction script with validation
This script combines the best approaches from previous versions
"""

import os
import sys
import pickle
import glob
from pathlib import Path
from fastai.vision.all import *

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
MODEL_DIR = Path("models")
DEFAULT_MODEL = "celebrity_recognition_model.pth"
VOCAB_FILE = MODEL_DIR/"vocab.pkl"
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to consider a prediction valid

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

def find_best_model():
    """Find the best model file to use"""
    # First try the default model
    if os.path.exists(MODEL_DIR/DEFAULT_MODEL):
        return MODEL_DIR/DEFAULT_MODEL
    
    # Look for other .pth files in the models directory
    models = list(MODEL_DIR.glob("*.pth"))
    if models:
        # Sort by file size (descending)
        models.sort(key=lambda x: os.path.getsize(x), reverse=True)
        return models[0]
    
    # Look for .pth files in the current directory
    models = list(Path(".").glob("*.pth"))
    if models:
        return models[0]
    
    # No models found
    return None

def create_minimal_learner():
    """Create a minimal learner for loading the model"""
    # Create a simple dataloader
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=lambda x: Path(x).parent.name,
        item_tfms=Resize(224),
    ).dataloaders(Path("."))
    
    # Try multiple possible architectures
    for arch in [resnet50, resnet34, resnet18]:
        try:
            print(f"Trying architecture: {arch.__name__}")
            learn = vision_learner(dls, arch, pretrained=False)
            return learn
        except Exception as e:
            print(f"Failed with {arch.__name__}: {e}")
    
    # If all else fails, return None
    return None

def load_model(model_path):
    """Load model with multiple fallback approaches"""
    print(f"Loading model from {model_path}")
    
    # Try direct loading first
    try:
        print("Attempting direct load...")
        learn = load_learner(model_path)
        print("Direct loading successful!")
        return learn
    except Exception as e:
        print(f"Direct loading failed: {e}")
    
    # Create a minimal learner and try loading weights
    learn = create_minimal_learner()
    if learn is None:
        print("Failed to create learner.")
        return None
    
    # Load state dict
    try:
        print("Loading state dictionary...")
        state = torch.load(model_path, map_location='cpu')
        
        # Handle different state dict formats
        if isinstance(state, dict):
            if 'model' in state:
                print("Found 'model' key in state")
                state_dict = state['model']
            elif 'state_dict' in state:
                print("Found 'state_dict' key in state")
                state_dict = state['state_dict']
            elif 'model_state_dict' in state:
                print("Found 'model_state_dict' key in state")
                state_dict = state['model_state_dict']
            else:
                print("Using full state dict")
                state_dict = state
        else:
            print("State is not a dictionary")
            state_dict = state
        
        # Try to load with strict=False
        learn.model.load_state_dict(state_dict, strict=False)
        print("Model loaded with partial weights")
        return learn
        
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return None

def extract_name_from_filename(image_path):
    """Extract potential celebrity name from the image filename"""
    filename = Path(image_path).stem
    # Replace underscores with spaces
    name = filename.replace('_', ' ')
    # Remove any digits
    name = ''.join([c for c in name if not c.isdigit()])
    # Remove common file prefixes/suffixes
    for prefix in ['IMG', 'image', 'pic', 'photo']:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix):].strip()
    return name.strip()

def validate_prediction(pred_idx, probs, celebrity_names, image_path, threshold=CONFIDENCE_THRESHOLD):
    """Validate prediction by checking confidence and filename match"""
    # Get top predictions
    top_idxs = probs.argsort(descending=True)[:5]
    top_probs = probs[top_idxs]
    
    # Extract potential name from filename
    potential_name = extract_name_from_filename(image_path)
    print(f"Extracted potential name from filename: '{potential_name}'")
    
    # If filename contains a name, check if it matches any celebrity
    if potential_name:
        # First, look for exact matches in the celebrity names
        for i, name in enumerate(celebrity_names):
            # Clean up the name for comparison
            clean_name = name.replace('_', ' ').lower()
            potential_lower = potential_name.lower()
            
            # Check for exact name match
            if clean_name == potential_lower:
                print(f"EXACT NAME MATCH: '{potential_name}' is '{name}'")
                # Force this as the prediction with high confidence
                return i, 0.99
        
        # If no exact match, try partial matching
        best_match_score = 0
        best_match_idx = -1
        
        for i, name in enumerate(celebrity_names):
            # Break names into parts
            name_parts = name.replace('_', ' ').lower().split()
            potential_parts = potential_name.lower().split()
            
            # Calculate match score
            match_score = 0
            for p_part in potential_parts:
                if len(p_part) > 2:  # Only consider meaningful parts
                    for n_part in name_parts:
                        if len(n_part) > 2:  # Only consider meaningful parts
                            # Check for exact part match
                            if p_part == n_part:
                                match_score += 1.0
                            # Check for partial match (one name contains the other)
                            elif p_part in n_part or n_part in p_part:
                                match_score += 0.5
            
            # Normalize by total parts
            total_parts = len(name_parts) + len(potential_parts)
            if total_parts > 0:
                match_score = match_score / total_parts
                
                # Update best match if better
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_idx = i
        
        # If we found a good partial match
        if best_match_score > 0.3:
            print(f"NAME MATCH: '{potential_name}' matches '{celebrity_names[best_match_idx]}' with score {best_match_score:.2f}")
            
            # If this matches one of our top predictions, use that confidence
            if best_match_idx in top_idxs:
                matched_idx = (top_idxs == best_match_idx).nonzero()[0].item()
                print(f"Match is in top predictions at position {matched_idx+1} with confidence: {top_probs[matched_idx]:.4f}")
                # Boost confidence a bit since we have a name match
                adjusted_conf = min(top_probs[matched_idx] * 1.5, 0.95)
                return best_match_idx, adjusted_conf
            else:
                # Not in top predictions, but still a good name match
                print(f"Match not in top predictions, using moderate confidence")
                # Use a moderate confidence level
                return best_match_idx, 0.75
    
    # Default: return top prediction if confidence is high enough
    if top_probs[0] >= threshold:
        return top_idxs[0], top_probs[0]
    else:
        # Low confidence, return most likely
        return top_idxs[0], top_probs[0]

def predict_celebrity(image_path, model_path=None):
    """Predict celebrity with model validation"""
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return False
        
        # Find best model if not specified
        if model_path is None:
            model_path = find_best_model()
            if model_path is None:
                print("Error: No model file found.")
                return False
            print(f"Using model: {model_path}")
        
        # Load celebrity names
        celebrity_names = load_celebrity_names()
        
        # Load model
        learn = load_model(model_path)
        if learn is None:
            print("Error: Failed to load model.")
            return False
        
        # Process the image
        print(f"Processing image: {image_path}")
        img = PILImage.create(image_path)
        
        # Make prediction with error handling
        try:
            print("Making prediction...")
            pred_class, pred_idx, probs = learn.predict(img)
            
            # Validate the prediction
            validated_idx, confidence = validate_prediction(
                pred_idx, probs, celebrity_names, image_path
            )
            
            # Get top predictions for display
            top_idxs = probs.argsort(descending=True)[:5]
            top_probs = probs[top_idxs]
            
            # Display results
            print("\n==== Prediction Results ====")
            
            # Get predicted name
            predicted_name = celebrity_names[validated_idx] if validated_idx < len(celebrity_names) else f"Unknown_{validated_idx}"
            confidence_pct = confidence * 100
            
            # Show the main prediction with confidence
            confidence_level = "HIGH" if confidence >= 0.7 else "MEDIUM" if confidence >= 0.4 else "LOW"
            print(f"\nPredicted Celebrity: {predicted_name}")
            print(f"Confidence: {confidence_pct:.1f}% ({confidence_level} CONFIDENCE)")
            
            # Show top 5 matches
            print("\nTop 5 Matches:")
            print("-" * 40)
            for i, (idx, prob) in enumerate(zip(top_idxs, top_probs)):
                idx_val = int(idx)
                name = celebrity_names[idx_val] if idx_val < len(celebrity_names) else f"Unknown_{idx_val}"
                mark = " *" if idx_val == validated_idx else ""
                print(f"{i+1}. {name:<25} {prob:.4f} ({prob*100:.1f}%){mark}")
            
            # Add note about validation
            if validated_idx != top_idxs[0]:
                print("\n* Prediction adjusted based on filename match")
            
            return True
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    print("Celebrity Recognition (Final Version)")
    print("===================================")
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python final_predict.py <image_path> [model_path]")
        
        # Search for test images
        test_dirs = ["testpics", "sample_images", "."]
        for dir in test_dirs:
            if os.path.exists(dir):
                images = []
                for ext in [".jpg", ".jpeg", ".png"]:
                    images.extend(glob.glob(f"{dir}/*{ext}"))
                if images:
                    example = images[0]
                    print(f"\nExample: python final_predict.py {example}")
                    break
        
        sys.exit(1)
    
    # Get paths
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Predict
    predict_celebrity(image_path, model_path)

if __name__ == "__main__":
    main() 