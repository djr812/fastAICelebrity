#!/usr/bin/env python
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
from torchvision import transforms
import sys

# Configuration
IMAGE_SIZE = 128
MODEL_PATH = 'models/pure_pytorch_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to predict from an image
def predict(img_path, vocab):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # Create model and load weights
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(vocab))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    # Get top 3 predictions
    probs_np = probs.cpu().numpy()[0]
    top3_indices = probs_np.argsort()[-3:][::-1]
    
    results = []
    for idx in top3_indices:
        if idx < len(vocab):
            pred_class = vocab[idx]
            confidence = float(probs_np[idx])
            results.append((pred_class, confidence))
    
    return results

# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pytorch_predict.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: Image path {img_path} does not exist")
        sys.exit(1)
    
    # Make prediction
    results = predict(img_path, dls.vocab)
    
    # Print results
    print("\nPredictions:")
    for i, (pred_class, confidence) in enumerate(results):
        print(f"{i+1}. {pred_class}: {confidence:.4f}")
