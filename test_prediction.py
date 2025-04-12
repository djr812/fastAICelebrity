#!/usr/bin/env python
"""
Test script for the improved_predict.py
This script runs the improved prediction and handles errors gracefully
"""

import os
import sys
import subprocess
from pathlib import Path

def fix_directory_structure():
    """Fix directory structure to ensure models are in the right place"""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create results directory
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Check if models/models exists and fix it
    if os.path.exists("models/models"):
        print("Found nested models/models directory, fixing...")
        # Try to move files up
        try:
            # List files in nested directory
            nested_files = os.listdir("models/models")
            for file in nested_files:
                src = os.path.join("models/models", file)
                dst = os.path.join("models", file)
                if os.path.isfile(src):
                    print(f"Moving {src} to {dst}")
                    os.rename(src, dst)
            # Remove nested directory if empty
            if not os.listdir("models/models"):
                os.rmdir("models/models")
        except Exception as e:
            print(f"Error fixing directory structure: {e}")
    
    # Check for model files in root directory and copy to models/
    for ext in [".pth", ".pkl"]:
        for file in Path(".").glob(f"*{ext}"):
            if os.path.isfile(file):
                dst = os.path.join("models", os.path.basename(file))
                if not os.path.exists(dst):
                    print(f"Copying {file} to models directory")
                    try:
                        import shutil
                        shutil.copy(file, dst)
                    except Exception as e:
                        print(f"Error copying file: {e}")

def list_available_models():
    """List available models"""
    print("\nAvailable models:")
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith((".pth", ".pkl")):
                size = os.path.getsize(os.path.join("models", file)) / (1024 * 1024)
                print(f"  {file} ({size:.2f} MB)")
    else:
        print("  No models directory found")

def run_improved_predict(image_path=None):
    """Run the improved prediction script"""
    # Default to using a test image if none provided
    if not image_path:
        # Look for test images
        test_dirs = ["testpics", "sample_images", "."]
        for dir in test_dirs:
            if os.path.exists(dir):
                for ext in [".jpg", ".jpeg", ".png"]:
                    images = list(Path(dir).glob(f"*{ext}"))
                    if images:
                        image_path = str(images[0])
                        print(f"Using first found image: {image_path}")
                        break
                if image_path:
                    break
    
    if not image_path:
        print("No test image found. Please provide an image path.")
        return False
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
    
    # Find available models
    available_model = None
    if os.path.exists("models"):
        pth_models = list(Path("models").glob("*.pth"))
        if pth_models:
            # Prefer celebrity_recognition_model.pth if available
            if os.path.exists("models/celebrity_recognition_model.pth"):
                available_model = "models/celebrity_recognition_model.pth"
            else:
                available_model = str(pth_models[0])
    
    # Run the improved prediction script
    try:
        cmd = [sys.executable, "improved_predict.py", "--image", image_path]
        if available_model:
            cmd.extend(["--model", os.path.basename(available_model)])
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        return True
    except Exception as e:
        print(f"Error running improved_predict.py: {e}")
        return False

def main():
    """Main function"""
    print("Celebrity Recognition Test")
    print("=========================")
    
    # Fix directory structure
    fix_directory_structure()
    
    # List available models
    list_available_models()
    
    # Get image path from command line argument or use default
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run the improved prediction
    if run_improved_predict(image_path):
        print("\nPrediction completed successfully!")
    else:
        print("\nPrediction failed. Please check the errors above.")
        print("\nTry running directly with:")
        print("  python improved_predict.py --image <path_to_image> --model celebrity_recognition_model.pth")

if __name__ == "__main__":
    main() 