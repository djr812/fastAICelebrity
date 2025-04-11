#!/usr/bin/env python
"""
Improved prediction script for celebrity recognition
Uses models trained with the improved_train.py script
Includes confidence calibration and visualization
"""

import os
import sys
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
DEFAULT_MODEL = "improved_celebrity_model.pth"
NAMES_FILE = MODEL_PATH/"celebrity_names.txt"
MAPPING_FILE = MODEL_PATH/"class_mapping.txt"
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

def load_celebrity_names():
    """Load celebrity name mapping"""
    id_to_name = {}
    
    # Try dedicated names file first
    if NAMES_FILE.exists():
        print(f"Loading celebrity names from {NAMES_FILE}")
        with open(NAMES_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    class_id, name = parts[0], parts[1]
                    id_to_name[class_id] = name
    
    # If we still don't have names, load from class mapping
    if not id_to_name and MAPPING_FILE.exists():
        print(f"Loading class mapping from {MAPPING_FILE}")
        with open(MAPPING_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    class_id, idx = parts[0], parts[1]
                    id_to_name[idx] = f"Celebrity_{class_id}"
    
    # Fallback to generic names from celebrities.txt
    if not id_to_name and os.path.exists("celebrities.txt"):
        print("Loading names from celebrities.txt")
        with open("celebrities.txt", "r") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
            for i, name in enumerate(names):
                id_to_name[str(i)] = name
    
    # Final fallback to generic names
    if not id_to_name:
        print("No name mapping found, using generic names")
        for i in range(100):
            id_to_name[str(i)] = f"Celebrity_{i}"
    
    # Create reverse mapping (index to name)
    idx_to_name = {}
    for i in range(len(id_to_name)):
        idx_to_name[i] = id_to_name.get(str(i), f"Celebrity_{i}")
    
    return idx_to_name

def create_model(checkpoint):
    """Create model from checkpoint"""
    # Determine number of classes
    num_classes = checkpoint.get('num_classes', 100)
    if 'num_classes' not in checkpoint and 'model_state_dict' in checkpoint:
        # Try to determine from final layer
        state_dict = checkpoint['model_state_dict']
        fc_layers = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
        if fc_layers:
            fc_layer = state_dict[fc_layers[-1]]
            if isinstance(fc_layer, torch.Tensor):
                num_classes = fc_layer.shape[0]
    
    print(f"Creating model with {num_classes} output classes")
    
    # Determine architecture
    model_name = checkpoint.get('architecture', 'resnet50')
    
    # Create model based on architecture
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:  # Default to resnet50
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        # Check if our model uses dropout
        if any('fc.0' in k for k in checkpoint['model_state_dict'].keys()):
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)
    
    return model, num_classes

def load_model(model_path):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading model from {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        model, num_classes = create_model(checkpoint)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading directly
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.eval()
        
        return model, num_classes
    
    except Exception as e:
        print(f"Error loading model: {e}")
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

def predict_image(model, image_path, idx_to_name, threshold=CONFIDENCE_THRESHOLD, save_output=False):
    """Predict celebrity from image with calibrated confidence"""
    try:
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
        celebrity_name = idx_to_name.get(top_idx, f"Unknown_{top_idx}")
        
        # Prepare results
        results = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            calibrated_prob = calibrate_confidence(float(prob))
            name = idx_to_name.get(int(idx), f"Unknown_{idx}")
            results.append({
                'name': name,
                'confidence': calibrated_prob,
                'raw_confidence': float(prob)
            })
        
        # Visualize results
        visualize_prediction(image, results, calibrated_confidence, threshold, save_output, image_path)
        
        return celebrity_name, calibrated_confidence, results
    
    except Exception as e:
        print(f"Error predicting image: {e}")
        return "Error", 0.0, []

def visualize_prediction(image, results, top_confidence, threshold, save_output, image_path):
    """Create visualization of prediction results"""
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
                 f"{confidences[i]:.2%}", va='center')
    
    # Add threshold line
    plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.2f})')
    plt.legend()
    
    plt.tight_layout()
    
    # Save or display
    if save_output:
        output_path = os.path.splitext(image_path)[0] + "_prediction.png"
        plt.savefig(output_path)
        print(f"Prediction visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def print_prediction_result(celebrity, confidence, threshold, results):
    """Print formatted prediction results"""
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    
    if confidence >= threshold:
        print(f"✓ Identified as: {celebrity}")
        print(f"  Confidence:    {confidence:.2%}")
    else:
        print(f"× Low confidence prediction: {celebrity}")
        print(f"  Confidence: {confidence:.2%} (below threshold of {threshold:.2%})")
        print("  This person might not be in the training dataset.")
    
    print("\nTop 5 matches:")
    for i, result in enumerate(results[:5], 1):
        conf_symbol = "✓" if result['confidence'] >= threshold else "×"
        print(f"{i}. {conf_symbol} {result['name']} - {result['confidence']:.2%}")
    
    print("="*50)

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Load model
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = MODEL_PATH/model_path
    
    model, num_classes = load_model(model_path)
    
    # Load celebrity names
    idx_to_name = load_celebrity_names()
    print(f"Loaded {len(idx_to_name)} celebrity names")
    
    # Check mode
    if args.image:
        # Single image mode
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return 1
        
        # Make prediction
        celebrity, confidence, results = predict_image(
            model, args.image, idx_to_name, args.threshold, args.save
        )
        
        # Print results
        print_prediction_result(celebrity, confidence, args.threshold, results)
    else:
        # Interactive mode
        print("\nCelebrity Recognition - Interactive Mode")
        print("Enter 'q' to quit")
        
        while True:
            # Get image path
            image_path = input("\nEnter image path: ")
            if image_path.lower() == 'q':
                break
            
            if not os.path.exists(image_path):
                print(f"Error: Image not found: {image_path}")
                continue
            
            # Make prediction
            celebrity, confidence, results = predict_image(
                model, image_path, idx_to_name, args.threshold, args.save
            )
            
            # Print results
            print_prediction_result(celebrity, confidence, args.threshold, results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 