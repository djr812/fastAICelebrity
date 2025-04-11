#!/usr/bin/env python
"""
Simple prediction script for celebrity recognition
This script focuses only on loading a model and making predictions
"""

import os
import sys
from pathlib import Path
import torch
from fastai.vision.all import *

# Configuration
MODEL_DIR = Path("models")
MODEL_NAMES = ["celebrity_recognition_model.pkl", "model.pth", "final_model.pth", "stage1.pth"]

def find_model():
    """Find a valid model file to load"""
    print("Looking for available model files...")
    
    # Check in the models directory first
    for name in MODEL_NAMES:
        model_path = MODEL_DIR/name
        if model_path.exists():
            print(f"Found model at {model_path}")
            return model_path
    
    # Check in the current directory
    for name in MODEL_NAMES:
        model_path = Path(name)
        if model_path.exists():
            print(f"Found model at {model_path}")
            return model_path
            
    return None

def load_model(model_path):
    """Load a trained model"""
    try:
        if str(model_path).endswith(".pkl"):
            print("Loading model using load_learner...")
            learn = load_learner(model_path)
            return learn
        else:
            print("Loading model weights...")
            # We need to create an empty learner first
            learn = cnn_learner(DataLoaders(), resnet50, pretrained=False)
            learn.load(str(model_path).replace(".pth", ""))
            return learn
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_celebrity(model, image_path):
    """Predict celebrity from image"""
    try:
        img = PILImage.create(image_path)
        pred_class, pred_idx, probabilities = model.predict(img)
        confidence = float(torch.max(probabilities))
        return pred_class, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0

def main():
    """Main prediction function"""
    print("Celebrity Recognition Prediction")
    print("===============================")
    
    # Find a valid model
    model_path = find_model()
    if model_path is None:
        print("ERROR: No model file found. Please train a model first.")
        return
    
    # Load the model
    model = load_model(model_path)
    if model is None:
        print("ERROR: Failed to load model.")
        return
    
    print("Model loaded successfully!")
    
    # Process images
    if len(sys.argv) > 1:
        # Process command line image
        image_path = sys.argv[1]
        celebrity, confidence = predict_celebrity(model, image_path)
        if celebrity:
            print(f"Predicted celebrity: {celebrity}")
            print(f"Confidence: {confidence:.2%}")
    else:
        # Interactive mode
        while True:
            image_path = input("\nEnter image path (or 'q' to quit): ")
            if image_path.lower() == 'q':
                break
                
            if not os.path.exists(image_path):
                print(f"Error: File {image_path} does not exist")
                continue
                
            celebrity, confidence = predict_celebrity(model, image_path)
            if celebrity:
                print(f"Predicted celebrity: {celebrity}")
                print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main() 