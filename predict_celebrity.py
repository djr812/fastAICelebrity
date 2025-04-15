#!/usr/bin/env python
import argparse
from pathlib import Path
from fastai.vision.all import *
import torch

def load_model(model_path):
    """Load the trained model."""
    print(f"Loading model from {model_path}...")
    learn = load_learner(model_path)
    return learn

def predict_celebrity(model, image_path):
    """Make a prediction on a single image."""
    print(f"Analyzing image: {image_path}")
    img = PILImage.create(image_path)
    pred_class, pred_idx, probs = model.predict(img)
    return pred_class, probs[pred_idx]

def main():
    parser = argparse.ArgumentParser(description='Predict celebrity from image')
    parser.add_argument('--model', type=str, default='models/celebrity_recognition_model.pkl',
                        help='Path to the trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image to predict')
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)
    
    # Make prediction
    celebrity, confidence = predict_celebrity(model, args.image)
    print(f"\nPrediction: {celebrity}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main() 