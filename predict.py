#!/usr/bin/env python
import os
import sys
import json
from celebrity_recognition import predict_with_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        print("If model_path is not provided, will use the default model")
        return
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found")
        return
        
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'models/celebrity_recognition_model.pth'
    
    # Make prediction
    result = predict_with_file(model_path, image_path)
    
    # Print result
    print(json.dumps(result, indent=2))
    
    # Print top 3 in a more readable format
    if 'top3' in result:
        print("\nTop 3 predictions:")
        for i, pred in enumerate(result['top3']):
            print(f"{i+1}. {pred['class']}: {pred['confidence']:.4f}")

if __name__ == "__main__":
    main()
