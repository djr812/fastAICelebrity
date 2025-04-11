#!/usr/bin/env python
import os
import tempfile
import base64
from pathlib import Path
from PIL import Image
import io
import pandas as pd
from fastai.vision.all import *
import gradio as gr

# Configuration
MODEL_PATH = Path("models")
MODEL_NAME = "celebrity_recognition_model.pkl"
IDENTITY_FILE = "list_identity_celeba.txt"
TOP_N = 5

# Load celebrity data
def load_identity_data():
    """Load celebrity identity data for display purposes."""
    try:
        df = pd.read_csv(IDENTITY_FILE, sep="\s+", skiprows=1, 
                        names=["image_id", "identity_name"])
        # Get unique celebrities for statistics
        unique_celebs = df['identity_name'].unique()
        return df, unique_celebs
    except Exception as e:
        print(f"Error loading identity data: {e}")
        return None, []

# Load model
def load_model():
    """Load the trained celebrity recognition model."""
    model_file = MODEL_PATH/MODEL_NAME
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}")
    
    learn = load_learner(model_file)
    return learn

def predict_celebrity(img, model):
    """Predict the celebrity identity from an image."""
    # Convert to format expected by model
    img_tensor = PILImage.create(img)
    
    # Get prediction
    prediction, _, probs = model.predict(img_tensor)
    
    # Get top N predictions
    top_probs, top_idxs = probs.topk(min(TOP_N, len(probs)))
    
    # Create result dictionary
    results = {}
    for i, (prob, idx) in enumerate(zip(top_probs, top_idxs)):
        celeb_name = model.dls.vocab[idx]
        results[celeb_name] = float(prob)
    
    # Sort by probability
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    
    return sorted_results

def create_interface():
    """Create the Gradio interface for the app."""
    # Load model
    try:
        model = load_model()
        df, unique_celebs = load_identity_data()
        
        # Initialize interface
        with gr.Blocks(title="Celebrity Face Recognition") as interface:
            gr.Markdown("# Celebrity Face Recognition")
            gr.Markdown(f"This model can recognize {len(unique_celebs)} different celebrities.")
            gr.Markdown("Upload a photo to see if the model can identify the celebrity.")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload Celebrity Image")
                    submit_btn = gr.Button("Identify Celebrity")
                
                with gr.Column():
                    output_label = gr.Label(label="Top Predictions")
            
            # Set up submission action
            submit_btn.click(
                fn=lambda img: predict_celebrity(img, model),
                inputs=input_image,
                outputs=output_label
            )
            
            # Examples
            gr.Markdown("## Tips")
            gr.Markdown("- For best results, use clear front-facing photos of celebrities")
            gr.Markdown("- The model was trained on aligned face images, so results may vary with different poses")
            
        return interface
    
    except Exception as e:
        print(f"Error creating interface: {e}")
        # Fallback minimal interface with error message
        with gr.Blocks(title="Error") as interface:
            gr.Markdown("# Error Loading Celebrity Recognition Model")
            gr.Markdown(f"Error: {str(e)}")
            gr.Markdown("Please ensure the model is trained and available at the correct location.")
        
        return interface

# Main application
if __name__ == "__main__":
    print("Starting Celebrity Recognition Web Interface...")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Check if model exists
    if not (MODEL_PATH/MODEL_NAME).exists():
        print(f"Model not found at {MODEL_PATH/MODEL_NAME}")
        print("Please train the model first using celebrity_recognition.py")
    
    # Launch the interface
    interface = create_interface()
    interface.launch(share=True)
    print("Interface closed.") 