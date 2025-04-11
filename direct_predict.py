#!/usr/bin/env python
"""
Direct prediction script that uses PyTorch directly
Bypassing FastAI loading issues
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Add the required safe globals for FastAI models
# This addresses the "Unsupported global" error in PyTorch 2.6+
try:
    # More comprehensive approach for safe globals
    import torch.serialization
    # Add all commonly used fastai globals
    safe_classes = ['L', 'fastai', 'Tensor', 'Module', 'collections', 'pd', 'np']
    for cls in safe_classes:
        try:
            torch.serialization.add_safe_globals([cls])
        except:
            pass
    print("Added safe globals for PyTorch")
except ImportError:
    print("Could not configure safe globals - older PyTorch version")

# Configuration
MODEL_PATHS = [
    'models/stage1.pth',
    'stage1.pth',
    'models/model.pth',
    'model.pth',
    'models/final_model.pth',
    'final_model.pth'
]

# Mapping file to convert indices to celebrity names
CELEBS_FILE = 'celebrities.txt'  # we'll create this from the model's vocabulary

# Prepare image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def find_model_file():
    """Find an available model file"""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None

def extract_vocab():
    """Create a mapping file from class indices to celebrity names"""
    vocab = []
    
    # First try to get actual celebrity identity data
    celeba_id_files = [
        'list_identity_celeba.txt',
        'identity_CelebA.txt',
        'identity_celeba.txt'
    ]
    
    # Try to load CelebA identity data first
    for id_file in celeba_id_files:
        if os.path.exists(id_file):
            print(f"Found CelebA identity file: {id_file}")
            try:
                # Read the identity file, which maps images to celebrity IDs
                with open(id_file, 'r') as f:
                    lines = f.readlines()
                
                # Skip header if present
                if len(lines) > 0 and not lines[0].strip().isdigit():
                    lines = lines[1:]
                
                # Extract unique celebrity IDs and count them
                celeb_ids = {}
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        celeb_id = parts[1]
                        if celeb_id not in celeb_ids:
                            celeb_ids[celeb_id] = 0
                        celeb_ids[celeb_id] += 1
                
                # Sort by frequency to get the most common ones
                sorted_celebs = sorted(celeb_ids.items(), key=lambda x: x[1], reverse=True)
                top_celebs = [f"CelebA_{cid}" for cid, _ in sorted_celebs[:100]]
                print(f"Loaded {len(top_celebs)} celebrity IDs from CelebA data")
                
                # If we also have a mapping file with actual names, use it
                if os.path.exists('celebrities.txt'):
                    with open('celebrities.txt', 'r') as f:
                        name_lines = f.readlines()
                    
                    name_list = [name.strip() for name in name_lines if name.strip()]
                    
                    # Create vocabulary with real names
                    if len(name_list) >= len(top_celebs):
                        vocab = name_list[:len(top_celebs)]
                        print(f"Mapped celebrity IDs to {len(vocab)} real names")
                        return vocab
                
                # If we don't have real names, return the IDs
                return top_celebs
            except Exception as e:
                print(f"Error parsing CelebA identity file: {e}")
    
    # Look for a vocabulary file next
    vocab_files = ['vocab.txt', 'models/vocab.txt', 'celebrities.txt', 'models/celebrities.txt']
    
    for file in vocab_files:
        if os.path.exists(file):
            print(f"Found vocabulary file: {file}")
            with open(file, 'r') as f:
                vocab = [line.strip() for line in f.readlines() if line.strip()]
            return vocab
            
    # If we don't have a vocabulary file, create a simple one with indices
    print("No vocabulary file found. Using generic class names.")
    return [f"Celebrity_{i}" for i in range(100)]

def create_fresh_model():
    """Create a fresh ResNet34 model"""
    # Create a ResNet model
    model = models.resnet34(weights=None)
    
    # Determine number of classes and modify the final layer
    num_classes = 100  # Match number of classes in celebrities.txt
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def simplified_load_model(model_path):
    """Simplified model loading approach that avoids complex serialization"""
    print(f"Loading model weights from {model_path} using simplified approach...")
    
    # Create fresh model
    model = create_fresh_model()
    
    try:
        # Try to load using basic approach
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check if it's a FastAI model and extract state dict
        if isinstance(checkpoint, dict):
            print("Loaded checkpoint dictionary")
            
            # Look for model key (FastAI format)
            if 'model' in checkpoint:
                print("Found 'model' key in checkpoint")
                state_dict = checkpoint['model']
            else:
                # Try other common keys
                for key in ['state_dict', 'model_state_dict', 'net', 'netG', 'weights']:
                    if key in checkpoint:
                        print(f"Found '{key}' in checkpoint")
                        state_dict = checkpoint[key]
                        break
                else:
                    # If no recognized keys, assume the whole thing is the state dict
                    print("Using entire checkpoint as state dict")
                    state_dict = checkpoint
        else:
            print("Checkpoint is not a dictionary, cannot extract state dict")
            return None
        
        # Clean up state dict to match model structure
        clean_dict = {}
        for k, v in state_dict.items():
            # Skip any problematic items
            if not isinstance(k, str):
                print(f"Skipping non-string key: {k}")
                continue
                
            # Remove common prefixes
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('encoder.'):
                k = k[8:]
                
            # Only keep items that look like model weights
            if ('weight' in k or 'bias' in k or 
                'running_mean' in k or 'running_var' in k or
                'num_batches_tracked' in k):
                clean_dict[k] = v
        
        # Load state dict with loose matching
        model.load_state_dict(clean_dict, strict=False)
        print(f"Loaded {len(clean_dict)} parameters into model")
        
        return model
    except Exception as e:
        print(f"Error in simplified loading: {e}")
        # If all else fails, just return a fresh untrained model
        print("Returning untrained model")
        return model

def predict_image(model, image_path, vocab):
    """Predict using the model"""
    try:
        # Load and preprocess the image
        print(f"Processing image: {image_path}")
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Set the model to evaluation mode
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top prediction
        top_prob, top_class = torch.max(probabilities, 0)
        
        # Ensure class index is within vocab range
        class_idx = min(top_class.item(), len(vocab)-1)
        
        # Get top 5 predictions (or fewer if not enough classes)
        k = min(5, len(vocab), len(probabilities))
        top5_probs, top5_indices = torch.topk(probabilities, k)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Top prediction: {vocab[class_idx]}, Confidence: {top_prob.item():.2%}")
        
        print("\nTop predictions:")
        for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
            idx_val = min(idx.item(), len(vocab)-1)  # Ensure index is in range
            print(f"{i+1}. {vocab[idx_val]}: {prob.item():.2%}")
            
        return True
    except Exception as e:
        print(f"Error making prediction: {e}")
        return False

def main():
    """Main function"""
    print("Direct Celebrity Recognition")
    print("===========================")
    
    # Find a model file
    model_path = find_model_file()
    if model_path is None:
        print("Error: No model file found. Please train a model first.")
        return
        
    print(f"Found model file: {model_path}")
    
    # Load the model using the simplified approach
    model = simplified_load_model(model_path)
    if model is None:
        print("Error: Failed to load model. Using an untrained model.")
        model = create_fresh_model()
        
    # Load or create vocabulary
    vocab = extract_vocab()
    print(f"Loaded {len(vocab)} celebrity names")
    
    # Check args
    if len(sys.argv) < 2:
        print("Usage: python direct_predict.py <image_path>")
        print("\nNo image provided. Entering interactive mode...")
        
        # Interactive mode
        while True:
            img_path = input("\nEnter an image path (or 'q' to quit): ")
            if img_path.lower() == 'q':
                break
                
            if not os.path.exists(img_path):
                print(f"Error: File not found: {img_path}")
                continue
                
            predict_image(model, img_path, vocab)
    else:
        # Use command line argument
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"Error: File not found: {img_path}")
            return
            
        # Make prediction
        predict_image(model, img_path, vocab)

if __name__ == "__main__":
    main() 