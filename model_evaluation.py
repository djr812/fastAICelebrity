#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from fastai.vision.all import *

# Configuration
DATA_DIR = Path("img_align_celeba")
IDENTITY_FILE = "list_identity_celeba.txt"
MODEL_PATH = Path("models")
MODEL_NAME = "celebrity_recognition_model.pkl"
RESULTS_DIR = Path("evaluation_results")
TOP_N = 5  # For top-N accuracy

def load_model():
    """Load the trained model."""
    # Create a list of possible model files to try
    possible_model_files = [
        MODEL_PATH/MODEL_NAME,           # Default .pkl file
        MODEL_PATH/'final_model.pth',    # .pth file in models directory
        Path('final_model.pth'),         # Direct path to .pth file
        Path('models/model.pth'),        # Explicitly nested path
        MODEL_PATH/'model.pth',          # Another common name
    ]
    
    print("Trying to load model from available files...")
    
    # Try each possible model file
    for model_file in possible_model_files:
        if model_file.exists():
            print(f"Found model file: {model_file}")
            try:
                # Try to load the model
                if str(model_file).endswith('.pkl'):
                    # Use load_learner for .pkl files
                    return load_learner(model_file)
                else:
                    # For .pth files, need a more complex approach
                    # This should be implemented based on how the model was saved
                    print(f"Found .pth model at {model_file}, but need to use a Learner object")
                    # Will need additional setup code here
            except Exception as e:
                print(f"Error loading model from {model_file}: {e}")
                continue
    
    # If we got here, we couldn't load the model
    raise FileNotFoundError(f"Could not find a valid model file. Please ensure a model is saved in {MODEL_PATH}.")

def load_test_data(min_samples_per_celeb=5):
    """Load celebrity identity data and create a test set."""
    # Skip the first few lines which contain metadata
    df = pd.read_csv(IDENTITY_FILE, sep="\s+", skiprows=1, 
                     names=["image_id", "identity_name"])
    
    # Get celebrities with sufficient images for better evaluation
    celebs_with_samples = df['identity_name'].value_counts()[df['identity_name'].value_counts() >= min_samples_per_celeb]
    
    # Create a test set with multiple images per celebrity for more reliable evaluation
    test_data = []
    for celeb in celebs_with_samples.index:
        celeb_images = df[df['identity_name'] == celeb]['image_id'].values
        # Randomly select multiple images for testing (20% of available images)
        num_test_imgs = max(1, min(10, int(len(celeb_images) * 0.2)))
        test_imgs = random.sample(list(celeb_images), num_test_imgs)
        for test_img in test_imgs:
            test_data.append({"image_id": test_img, "identity_name": celeb})
    
    test_df = pd.DataFrame(test_data)
    print(f"Created test set with {len(test_df)} images from {len(celebs_with_samples)} celebrities")
    return test_df

def compute_top_n_accuracy(pred_probs, actual_labels, model, n=TOP_N):
    """Compute top-N accuracy."""
    correct = 0
    total = len(actual_labels)
    
    for i, probs in enumerate(pred_probs):
        # Get indices of top N probabilities
        top_n_indices = torch.topk(probs, n).indices
        # Convert to class names
        top_n_classes = [model.dls.vocab[idx] for idx in top_n_indices]
        # Check if actual label is in top N
        if actual_labels[i] in top_n_classes:
            correct += 1
    
    return correct / total

def evaluate_model(learn, test_df):
    """Evaluate the model on test data."""
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Lists to store results
    actual_labels = []
    predicted_labels = []
    confidence_scores = []
    pred_probabilities = []
    correct_predictions = 0
    
    # Evaluate each test image
    for i, row in test_df.iterrows():
        img_path = DATA_DIR / row['image_id']
        actual_celeb = row['identity_name']
        
        # Load and predict
        img = PILImage.create(img_path)
        pred_celeb, _, probs = learn.predict(img)
        confidence = float(torch.max(probs))
        
        # Record results
        actual_labels.append(actual_celeb)
        predicted_labels.append(pred_celeb)
        confidence_scores.append(confidence)
        pred_probabilities.append(probs)
        
        if pred_celeb == actual_celeb:
            correct_predictions += 1
        
        # Print progress
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(test_df)} test images")
    
    # Calculate basic accuracy
    accuracy = correct_predictions / len(test_df)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Calculate top-N accuracy
    top_n_acc = compute_top_n_accuracy(pred_probabilities, actual_labels, learn)
    print(f"Top-{TOP_N} Accuracy: {top_n_acc:.4f}")
    
    # Calculate more detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_labels, predicted_labels, average='weighted'
    )
    
    # Print results summary
    print("\nDetailed Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate per-celebrity performance
    celebrity_metrics = {}
    for celeb in set(actual_labels):
        # Get indices where this celebrity is the actual label
        indices = [i for i, label in enumerate(actual_labels) if label == celeb]
        if indices:
            # Calculate accuracy for this celebrity
            celeb_acc = sum(1 for i in indices if predicted_labels[i] == celeb) / len(indices)
            celeb_count = len(indices)
            celebrity_metrics[celeb] = {'accuracy': celeb_acc, 'count': celeb_count}
    
    # Get top and bottom performing celebrities
    sorted_celebs = sorted(celebrity_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\nTop 5 Best Recognized Celebrities:")
    for celeb, metrics in sorted_celebs[:5]:
        print(f"{celeb}: {metrics['accuracy']:.2%} accuracy ({metrics['count']} samples)")
    
    print("\nTop 5 Worst Recognized Celebrities:")
    for celeb, metrics in sorted_celebs[-5:]:
        print(f"{celeb}: {metrics['accuracy']:.2%} accuracy ({metrics['count']} samples)")
    
    # Generate and save classification report
    report = classification_report(actual_labels, predicted_labels)
    with open(RESULTS_DIR/'classification_report.txt', 'w') as f:
        f.write(f"Overall Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Top-{TOP_N} Accuracy: {top_n_acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Classification report saved to {RESULTS_DIR/'classification_report.txt'}")
    
    # Generate confusion matrix for top celebrities
    top_celebs = pd.Series(actual_labels).value_counts().head(15).index
    top_actual = [label if label in top_celebs else 'Other' for label in actual_labels]
    top_predicted = [label if label in top_celebs else 'Other' for label in predicted_labels]
    
    cm = confusion_matrix(top_actual, top_predicted, labels=list(top_celebs) + ['Other'])
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=list(top_celebs) + ['Other'],
               yticklabels=list(top_celebs) + ['Other'])
    plt.title('Confusion Matrix (Top 15 Celebrities)')
    plt.tight_layout()
    plt.ylabel('True Celebrity')
    plt.xlabel('Predicted Celebrity')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(RESULTS_DIR/'confusion_matrix.png')
    print(f"Confusion matrix saved to {RESULTS_DIR/'confusion_matrix.png'}")
    
    # Generate confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(confidence_scores, bins=20, kde=True)
    plt.axvline(x=0.5, color='r', linestyle='--', label='50% confidence')
    plt.axvline(x=0.7, color='g', linestyle='--', label='70% confidence')
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(RESULTS_DIR/'confidence_distribution.png')
    print(f"Confidence distribution saved to {RESULTS_DIR/'confidence_distribution.png'}")
    
    # Plot accuracy vs confidence threshold
    thresholds = np.arange(0.1, 1.0, 0.05)
    accuracies = []
    coverage = []
    
    for threshold in thresholds:
        # Filter predictions with confidence above threshold
        high_conf_indices = [i for i, conf in enumerate(confidence_scores) if conf >= threshold]
        if high_conf_indices:
            # Calculate accuracy for high confidence predictions
            high_conf_acc = sum(1 for i in high_conf_indices if predicted_labels[i] == actual_labels[i]) / len(high_conf_indices)
            accuracies.append(high_conf_acc)
            coverage.append(len(high_conf_indices) / len(confidence_scores))
        else:
            accuracies.append(0)
            coverage.append(0)
    
    # Plot accuracy vs threshold
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, accuracies, marker='o')
    plt.title('Accuracy vs Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot coverage vs threshold
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, coverage, marker='o', color='orange')
    plt.title('Coverage vs Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('% of Predictions Above Threshold')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/'accuracy_by_confidence.png')
    print(f"Accuracy vs confidence plot saved to {RESULTS_DIR/'accuracy_by_confidence.png'}")
    
    return accuracy, top_n_acc, precision, recall, f1

def main():
    """Main function to evaluate the celebrity recognition model."""
    print("Celebrity Recognition Model Evaluation")
    print("--------------------------------------")
    
    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    try:
        # Load model
        learn = load_model()
        
        # Load test data
        test_data = load_test_data()
        
        # Evaluate model
        accuracy, top_n_acc, precision, recall, f1 = evaluate_model(learn, test_data)
        
        print("\nEvaluation completed successfully!")
        print(f"Overall Results Summary:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Top-{TOP_N} Accuracy: {top_n_acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("\nFallback to direct prediction using improved_predict.py...")
        
        # Suggest using improved_predict as fallback
        print("You can use improved_predict.py for direct inference instead:")
        print("  python improved_predict.py --image [your_image_path]")

if __name__ == "__main__":
    main() 