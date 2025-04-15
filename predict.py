#!/usr/bin/env python
import os
import sys
import json
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import logging
import torch.nn as nn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
MODEL_PATH = Path("models")
MODEL_NAME = "celebrity_recognition_model.pth"
VOCAB_FILE = "models/vocab.pkl"
NUM_CLASSES = 6348  # Number of celebrities in our dataset
IMAGE_SIZE = 224

def load_vocab():
    """Load the vocabulary file."""
    try:
        import pickle
        if not os.path.exists(VOCAB_FILE):
            logging.error(f"Vocabulary file not found: {VOCAB_FILE}")
            return None
            
        with open(VOCAB_FILE, 'rb') as f:
            vocab = pickle.load(f)
            
        if not vocab or len(vocab) == 0:
            logging.error("Vocabulary is empty")
            return None
            
        logging.info(f"Loaded vocabulary with {len(vocab)} classes")
        return vocab
    except Exception as e:
        logging.error(f"Error loading vocabulary: {e}")
        return None

def create_model(num_classes):
    """Create a new model with the correct number of classes."""
    from torchvision.models import resnet50
    model = resnet50(pretrained=False)
    
    # Modify the final layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features),
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model

def load_model(model_path, num_classes):
    """Load the trained model."""
    try:
        logging.info(f"Loading model from {model_path}")
        
        # First create a model with 10 classes to match the saved state
        model = create_model(10)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Handle different state dict formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Load the state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        # Now create the final model with correct number of classes
        final_model = create_model(num_classes)
        
        # Copy the weights from the loaded model to the final model
        # for all layers except the last one
        for name, param in model.named_parameters():
            if 'fc.6' not in name:  # Skip the final layer
                final_model.state_dict()[name].copy_(param)
        
        final_model.eval()
        return final_model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def predict_image(model, image_path, vocab):
    """Make a prediction on a single image."""
    try:
        if vocab is None or len(vocab) == 0:
            return {'error': "Vocabulary not loaded or empty"}
            
        # Load and preprocess the image
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            
            # Get top 3 predictions
            values, indices = torch.topk(probs, min(3, len(vocab)))
            
            # Convert to lists
            values = values.numpy()[0]
            indices = indices.numpy()[0]
            
            # Create results dictionary
            results = {
                'prediction': vocab[indices[0]],
                'confidence': float(values[0]),
                'top3': [
                    {'class': vocab[idx], 'confidence': float(val)}
                    for val, idx in zip(values, indices)
                ]
            }
            
            return results
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return {'error': str(e)}

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        print("If model_path is not provided, will use the default model")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found")
        return
    
    model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH/MODEL_NAME
    
    # Load vocabulary
    vocab = load_vocab()
    if vocab is None:
        print("Error: Could not load vocabulary file")
        return
    
    # Load model
    model = load_model(model_path, len(vocab))
    if model is None:
        print("Error: Could not load model")
        return
    
    # Make prediction
    result = predict_image(model, image_path, vocab)
    
    # Print result
    print(json.dumps(result, indent=2))
    
    # Print top 3 in a more readable format
    if 'top3' in result:
        print("\nTop 3 predictions:")
        for i, pred in enumerate(result['top3']):
            print(f"{i+1}. {pred['class']}: {pred['confidence']:.4f}")

if __name__ == "__main__":
    main()
