#!/usr/bin/env python
"""
Improved prediction script for celebrity recognition
Uses models trained with the improved_train.py script
Includes confidence calibration and visualization
"""

import os
import sys
import pickle
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import models, transforms
import argparse

# Configuration
MODEL_PATH = Path("models")
DEFAULT_MODEL = "celebrity_recognition_model.pth"  # Changed to your existing model
VOCAB_FILE = MODEL_PATH/"vocab.pkl"
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence to consider a prediction valid

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Celebrity Recognition Prediction")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"Model file to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--save", action="store_true", help="Save visualization")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    return parser.parse_args()

def load_vocabulary():
    """Load vocabulary mapping from pickle file"""
    vocab = {}
    try:
        if VOCAB_FILE.exists():
            print(f"Loading vocabulary from {VOCAB_FILE}")
            with open(VOCAB_FILE, 'rb') as f:
                vocab = pickle.load(f)
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
    
    # Fallback to celebrities.txt if vocab is empty
    if not vocab and os.path.exists("celebrities.txt"):
        print("Loading names from celebrities.txt")
        with open("celebrities.txt", "r") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
            for i, name in enumerate(names):
                vocab[i] = name
    
    # If still no vocabulary, use generic names
    if not vocab:
        print("No vocabulary found, using generic names")
        for i in range(100):
            vocab[i] = f"Celebrity_{i}"
    
    return vocab

def create_model(num_classes=100, arch='resnet50'):
    """Create model architecture"""
    if arch == 'resnet18':
        model = models.resnet18(weights=None)
    elif arch == 'resnet34':
        model = models.resnet34(weights=None)
    else:  # Default to resnet50
        model = models.resnet50(weights=None)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def load_model(model_path):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading model from {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Determine number of classes
        num_classes = 100  # Default
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # This is a dictionary-based checkpoint
            state_dict = checkpoint['model_state_dict']
            architecture = checkpoint.get('architecture', 'resnet50')
            
            # Try to determine number of classes from the final layer
            fc_keys = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
            if fc_keys:
                fc_layer = state_dict[fc_keys[-1]]
                if isinstance(fc_layer, torch.Tensor):
                    num_classes = fc_layer.shape[0]
        else:
            # Direct state dict
            state_dict = checkpoint
            architecture = 'resnet50'  # Default
            
            # Try to determine number of classes
            fc_keys = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
            if fc_keys:
                fc_layer = state_dict[fc_keys[-1]]
                if isinstance(fc_layer, torch.Tensor):
                    num_classes = fc_layer.shape[0]
        
        print(f"Creating model with {num_classes} output classes using {architecture}")
        
        # Create model
        model = create_model(num_classes, architecture)
        
        # Load state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.eval()
        
        return model, num_classes
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Details: {str(e)}")
        sys.exit(1)

def calibrate_confidence(confidence, method='temperature', t=1.5):
    """Calibrate confidence scores to be more realistic"""
    if method == 'temperature':
        # Temperature scaling (higher T = lower confidence)
        return confidence ** (1 / t)
    elif method == 'sigmoid':
        # Sigmoid calibration
        return 1 / (1 + np.exp(-(12 * confidence - 6)))
    else:
        return confidence

def predict_image(model, image_path, vocab, threshold=CONFIDENCE_THRESHOLD, save_output=False):
    """Predict celebrity from image with calibrated confidence"""
    try:
        # Check if image path exists
        if not os.path.exists(image_path):
            return json.dumps({"error": f"Image file not found: {image_path}"}, indent=2)
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        # Convert tensors to numpy for easier handling
        top_probs = top_probs.numpy()
        top_indices = top_indices.numpy()
        
        # Get top prediction
        top_idx = int(top_indices[0])
        raw_confidence = float(top_probs[0])
        
        # Calibrate confidence
        calibrated_confidence = calibrate_confidence(raw_confidence)
        
        # Get prediction name
        celebrity_name = vocab.get(top_idx, f"Unknown_{top_idx}")
        
        # Prepare results
        results = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            calibrated_prob = calibrate_confidence(float(prob))
            name = vocab.get(int(idx), f"Unknown_{idx}")
            results.append({
                'name': name,
                'confidence': calibrated_prob,
                'raw_confidence': float(prob)
            })
        
        # Visualize results
        visualize_prediction(image, results, calibrated_confidence, threshold, save_output, image_path)
        
        return celebrity_name, calibrated_confidence, results
    
    except Exception as e:
        error_msg = {"error": f"PyTorch prediction error: {str(e)}"}
        print(json.dumps(error_msg, indent=2))
        return "Error", 0.0, []

def visualize_prediction(image, results, top_confidence, threshold, save_output, image_path):
    """Create visualization of prediction results"""
    try:
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Display image on the left
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Display confidence bars on the right
        plt.subplot(1, 2, 2)
        
        # Extract names and confidences
        names = [result['name'] for result in results]
        confidences = [result['confidence'] for result in results]
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(confidences)), confidences, color=['green' if c >= threshold else 'orange' for c in confidences])
        plt.yticks(range(len(names)), names)
        plt.xlabel('Confidence (calibrated)')
        plt.title('Predictions')
        plt.xlim(0, 1)
        
        # Add confidence values
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{confidences[i]:.2f}', va='center')
        
        plt.tight_layout()
        
        # Save or show visualization
        if save_output:
            output_dir = Path("predictions")
            output_dir.mkdir(exist_ok=True)
            
            # Get original filename
            original_name = Path(image_path).stem
            output_path = output_dir / f"{original_name}_prediction.png"
            
            plt.savefig(output_path)
            print(f"Saved prediction visualization to {output_path}")
        else:
            plt.show()
    
    except Exception as e:
        print(f"Error in visualization: {e}")

def print_prediction_result(celebrity, confidence, threshold, results):
    """Print prediction results in a nice format"""
    # Print header
    print("\n" + "="*50)
    print("Celebrity Recognition Results")
    print("="*50)
    
    # Print top result with confidence
    status = "✓ MATCH" if confidence >= threshold else "✗ LOW CONFIDENCE"
    print(f"\nPrediction: {celebrity} ({confidence:.2f}) {status}")
    
    # Print confidence threshold
    print(f"Confidence threshold: {threshold:.2f}")
    
    # Print table of top results
    print("\nTop matches:")
    print("-"*50)
    print(f"{'Name':<30} {'Confidence':<10} {'Raw Score':<10}")
    print("-"*50)
    
    for result in results:
        name = result['name']
        conf = result['confidence']
        raw = result['raw_confidence']
        print(f"{name:<30} {conf:.2f}{' *' if conf >= threshold else '':2} {raw:.2f}")
    
    print("="*50)
    print("* Matches above confidence threshold")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Resolve paths
    model_path = MODEL_PATH / args.model
    
    # Load model
    try:
        model, num_classes = load_model(model_path)
        print("Model loaded successfully")
        
        # Load vocabulary 
        vocab = load_vocabulary()
        
        # If no image provided, print help and exit
        if not args.image:
            print("Error: No image provided")
            print("Usage: python improved_predict.py --image <path_to_image>")
            sys.exit(1)
        
        # Make prediction
        celebrity, confidence, results = predict_image(
            model, args.image, vocab, args.threshold, args.save
        )
        
        # Print results
        if celebrity != "Error":
            print_prediction_result(celebrity, confidence, args.threshold, results)
        
    except Exception as e:
        error_msg = {"error": str(e)}
        print(json.dumps(error_msg, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main() 