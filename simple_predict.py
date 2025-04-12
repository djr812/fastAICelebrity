#!/usr/bin/env python
"""
Simple, direct prediction script for the celebrity recognition model.
Usage: python simple_predict.py <image_path>
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Fixed model path
MODEL_PATH = "models/celebrity_recognition_model.pth"
VOCAB_PATH = "models/vocab.pkl"

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    """Load the model with error handling"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    
    try:
        # Load model file
        print(f"Loading model from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        # Check if this is a FastAI model
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print("Detected FastAI model format")
            model_state_dict = checkpoint['model']
            # Check if we need to trim 'model.' prefix from keys
            if all(k.startswith('model.') for k in model_state_dict.keys()):
                print("Trimming 'model.' prefix from state dict keys")
                model_state_dict = {k[6:]: v for k, v in model_state_dict.items() if k.startswith('model.')}
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("Detected PyTorch Lightning or similar format")
            model_state_dict = checkpoint['state_dict']
            # Check if we need to trim module prefix
            if all(k.startswith('model.') for k in model_state_dict.keys()):
                model_state_dict = {k[6:]: v for k, v in model_state_dict.items()}
            elif all(k.startswith('module.') for k in model_state_dict.keys()):
                model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Detected standard checkpoint format")
            model_state_dict = checkpoint['model_state_dict']
        else:
            # Assume it's a direct state dict
            print("Assuming direct state dictionary format")
            model_state_dict = checkpoint
            
        # Try to determine the architecture 
        architecture = 'resnet50'  # Default
        if isinstance(checkpoint, dict):
            architecture = checkpoint.get('architecture', 'resnet50')
            
        # Try to determine number of classes from the state dict
        num_classes = 100  # Default
        fc_key = None
        
        # Look for fully connected layer weights to determine classes
        for k in model_state_dict.keys():
            if 'fc.weight' in k:
                fc_key = k
                break
            
        if fc_key and isinstance(model_state_dict[fc_key], torch.Tensor):
            num_classes = model_state_dict[fc_key].shape[0]
        
        print(f"Using {architecture} with {num_classes} classes")
        
        # Create model based on architecture
        if architecture == 'resnet18':
            model = models.resnet18(weights=None)
        elif architecture == 'resnet34':
            model = models.resnet34(weights=None)
        else:  # Default to resnet50
            model = models.resnet50(weights=None)
        
        # Update the final layer to match number of classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        # Try to load the state dict directly
        try:
            model.load_state_dict(model_state_dict)
        except Exception as e:
            print(f"Standard loading failed: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(model_state_dict, strict=False)
            print("Warning: Some parameters were not loaded. Model may not be fully accurate.")
        
        # Set to evaluation mode
        model.eval()
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        # Print more details about the exception for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

def load_vocab():
    """Load vocabulary for class names"""
    try:
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, 'rb') as f:
                vocab_data = pickle.load(f)
                # Check if vocab is a list or dict
                if isinstance(vocab_data, list):
                    print(f"Loaded vocabulary as list with {len(vocab_data)} items")
                    # Convert list to dictionary
                    return {i: name for i, name in enumerate(vocab_data)}
                elif isinstance(vocab_data, dict):
                    print(f"Loaded vocabulary as dictionary with {len(vocab_data)} items")
                    return vocab_data
                else:
                    print(f"Vocabulary is in unexpected format: {type(vocab_data)}")
    except Exception as e:
        print(f"Warning: Couldn't load vocabulary: {e}")
    
    # Try to load from celebrities.txt as fallback
    if os.path.exists("celebrities.txt"):
        try:
            vocab = {}
            with open("celebrities.txt", "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                for i, name in enumerate(lines):
                    vocab[i] = name
            print(f"Loaded {len(vocab)} names from celebrities.txt")
            return vocab
        except Exception as e:
            print(f"Error loading from celebrities.txt: {e}")
    
    # Default vocabulary (generic class names)
    print("Using default generic class names")
    return {i: f"Celebrity_{i}" for i in range(100)}

def predict(model, image_path, vocab):
    """Make a prediction on an image"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    try:
        # Load and preprocess image
        print(f"Processing image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        # Print results
        print("\n==== Prediction Results ====")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            idx_int = int(idx)
            # Handle both list and dictionary formats
            if isinstance(vocab, dict):
                name = vocab.get(idx_int, f"Unknown_{idx_int}")
            else:
                # Fallback for unexpected format
                name = f"Celebrity_{idx_int}"
            print(f"{i+1}. {name}: {prob.item():.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python simple_predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load model and vocabulary
    model = load_model()
    vocab = load_vocab()
    
    # Make prediction
    predict(model, image_path, vocab)

if __name__ == "__main__":
    main() 