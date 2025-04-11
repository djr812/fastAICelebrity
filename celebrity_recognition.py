#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch
from fastai.vision.all import *
from fastai.callback.schedule import minimum, steep  # Add missing imports for LR finder
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import torch.serialization
import sys

# Configuration
DATA_DIR = Path("img_align_celeba")
IDENTITY_FILE = "list_identity_celeba.txt"
MODEL_PATH = Path("models")
RESULTS_DIR = Path("evaluation_results")
MODEL_NAME = "celebrity_recognition_model.pkl"
BATCH_SIZE = 12  # Smaller batch size to avoid memory issues
IMAGE_SIZE = 224  # Increased image size for better feature extraction
FINAL_IMAGE_SIZE = 256  # Final fine-tuning size for better detail capture
NUM_EPOCHS = 5  # First phase epochs
FINE_TUNE_EPOCHS = 10  # More epochs for fine-tuning
LEARNING_RATE = 1e-3  # Start with higher learning rate and let LR finder tune it
MIN_IMAGES_PER_CELEB = 15  # Increased minimum to ensure better quality training data
MAX_CELEBRITIES = 50  # Focus on fewer celebrities to improve accuracy
USE_MIXUP = True
MIXUP_ALPHA = 0.4  # Mixup alpha parameter for training robustness
LABEL_SMOOTHING = 0.1
USE_PROGRESSIVE_RESIZING = True  # Use progressive resizing for better training

# Define functions at the module level to avoid pickling issues
def get_x(r): return DATA_DIR/r['image_id']
def get_y(r): return r['identity_name']

def load_identity_data():
    """Load celebrity identity data from the text file."""
    print("Loading identity data...")
    # Skip the first few lines which contain metadata
    df = pd.read_csv(IDENTITY_FILE, sep=r"\s+", skiprows=1, 
                     names=["image_id", "identity_name"])
    print(f"Loaded {len(df)} image identity records")
    
    # Limit to top MAX_CELEBRITIES with most images for faster training
    if MAX_CELEBRITIES > 0:
        top_celebs = df['identity_name'].value_counts().head(MAX_CELEBRITIES).index
        df = df[df['identity_name'].isin(top_celebs)]
        print(f"Limited data to top {MAX_CELEBRITIES} celebrities ({len(df)} images)")
    
    return df

def setup_data_loaders(df, img_size=IMAGE_SIZE):
    """Set up FastAI data loaders for training."""
    # Make sure model directory exists
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Filter out celebrities with fewer than MIN_IMAGES_PER_CELEB images for better training
    celebrity_counts = df['identity_name'].value_counts()
    valid_celebrities = celebrity_counts[celebrity_counts >= MIN_IMAGES_PER_CELEB].index
    filtered_df = df[df['identity_name'].isin(valid_celebrities)].copy()
    
    print(f"Filtered out {len(df) - len(filtered_df)} images with too few examples per celebrity")
    print(f"Proceeding with {len(filtered_df)} images of {len(valid_celebrities)} celebrities")
    
    # Split data into train and validation sets
    train_df, valid_df = train_test_split(
        filtered_df, test_size=0.2, stratify=filtered_df['identity_name'], random_state=42
    )
    
    # Reset indices for cleaner dataframes
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(valid_df)} images")
    
    # Create class weights for handling imbalance
    unique_labels = filtered_df['identity_name'].unique()
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=filtered_df['identity_name']
    )
    class_weight_dict = dict(zip(unique_labels, class_weights))
    print(f"Applied class weighting to handle imbalanced data")
    
    # Enhanced data augmentation for improved generalization
    # More aggressive augmentation to improve model robustness
    tfms = [
        *aug_transforms(
            do_flip=True, 
            flip_vert=False, 
            max_rotate=25.0,  # Increased rotation
            max_zoom=1.2,     # More zoom variation
            max_lighting=0.4, # More lighting variation
            max_warp=0.25,    # More warping
            p_affine=0.85,    # Higher probability of applying affine transforms
            p_lighting=0.85,  # Higher probability of applying lighting changes
        ),
        Normalize.from_stats(*imagenet_stats)
    ]
    
    # Add additional augmentation for more robustness
    if USE_MIXUP:
        # Use RandomErasing with compatible parameters for the current FastAI version
        tfms.append(RandomErasing(p=0.7, max_count=6))
        # Add more rotation for additional robustness instead of Cutout
        tfms.append(Rotate(max_deg=15, p=0.8))
    
    # Create FastAI DataBlock for handling image data
    celeb_data = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=get_x,
        get_y=get_y,
        splitter=ColSplitter(),
        item_tfms=[
            Resize(img_size, method=ResizeMethod.Squish),
            RandomResizedCrop(img_size, min_scale=0.85, ratio=(0.9, 1.1))
        ],
        batch_tfms=tfms
    )
    
    # Add a column for splitting the data
    filtered_df['is_valid'] = np.where(filtered_df['image_id'].isin(valid_df['image_id']), True, False)
    
    # Create dataloaders
    dls = celeb_data.dataloaders(filtered_df, bs=BATCH_SIZE)
    
    # Store class weights for use in loss function
    dls.class_weights = torch.FloatTensor([class_weight_dict[c] for c in dls.vocab])
    
    return dls

# Define a helper function for safely loading models
def safe_load_model(learner, name):
    """Safely load a model with PyTorch 2.6+ compatibility."""
    try:
        # First try the normal loading
        learner.load(name)
        print(f"Successfully loaded model {name} with normal loading")
    except Exception as e1:
        print(f"Error loading model with normal method: {e1}")
        try:
            # Try with weights_only=False for PyTorch 2.6+
            file = f"{name}.pth"
            if not Path(file).exists():
                # Try models directory
                file = Path('models')/f"{name}.pth"
                if not file.exists():
                    raise FileNotFoundError(f"Could not find model file for {name}")
                
            # Determine the correct device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Attempting to load with weights_only=False from {file} to device {device}")
            
            # Add safe globals for PyTorch 2.6+
            torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
            state = torch.load(file, map_location=torch.device(device), weights_only=False)
            
            # Apply the state to the model
            learner.model.load_state_dict(state['model'], strict=False)
            print(f"Successfully loaded model weights for {name}")
            
            # Explicitly move the model to the correct device
            learner.model.to(device)
            print(f"Moved model to {device}")
            
            # Try to load optimizer state if needed
            if hasattr(learner, 'opt') and learner.opt is not None and 'opt' in state:
                try:
                    # Create a clean optimizer first
                    learner.create_opt()
                    # Then load the state
                    learner.opt.load_state_dict(state['opt'])
                    # Make sure optimizer state is on the same device as the model
                    for param_group in learner.opt.param_groups:
                        for param in param_group['params']:
                            if param.device != learner.model.parameters().__next__().device:
                                param.data = param.data.to(device)
                                if param.grad is not None:
                                    param.grad.data = param.grad.data.to(device)
                    print("Loaded and aligned optimizer state")
                except Exception as e:
                    print(f"Could not load optimizer state: {e}")
                    print("Continuing with new optimizer")
                    # Create a fresh optimizer
                    learner.create_opt()
            else:
                # Create a fresh optimizer to avoid device mismatch issues
                learner.create_opt()
                print("Created new optimizer")
                    
            return True
                    
        except Exception as e2:
            print(f"Error loading model with weights_only=False: {e2}")
            try:
                # Last resort: manual loading of just the model weights
                file = f"{name}.pth"
                if not Path(file).exists():
                    # Try models directory
                    file = Path('models')/f"{name}.pth"
                    
                # Use a custom unpickling approach
                print(f"Attempting manual load from {file}")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                state_dict = None
                
                try:
                    # Try to load the entire state dict first with simple torch load
                    state_dict = torch.load(file, map_location=torch.device(device), weights_only=True)
                except:
                    # Fall back to manual unpickling
                    with open(file, 'rb') as f:
                        import pickle, io
                        unpickler = pickle._Unpickler(f)
                        unpickler.find_class = lambda mod, name: getattr(sys.modules.get(mod, type('missing', (), {})), name)
                        state_dict = unpickler.load()
                
                # Extract just the model weights
                if isinstance(state_dict, dict) and 'model' in state_dict:
                    learner.model.load_state_dict(state_dict['model'], strict=False)
                    learner.model.to(device)  # Ensure model is on the correct device
                    print(f"Successfully loaded model weights manually for {name} to {device}")
                    # Create fresh optimizer
                    learner.create_opt()
                    return True
                else:
                    print("Could not find model weights in saved state")
                    return False
                    
            except Exception as e3:
                print(f"All loading methods failed: {e3}")
                return False
    return True

def train_model(dls, dls_final=None):
    """Train the celebrity recognition model using basic methods that work across FastAI versions."""
    print("Training model...")
    
    # Determine the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable memory efficient methods
    torch.backends.cudnn.benchmark = True
    
    # Create a CNN learner with more advanced architecture
    learn = vision_learner(
        dls, 
        'resnet50',  # Use string parameter for better compatibility
        metrics=[error_rate, accuracy],
        loss_func=LabelSmoothingCrossEntropy(eps=LABEL_SMOOTHING),  # Use label smoothing
        pretrained=True,  # Ensure we're using pretrained weights
        normalize=True    # Normalize inputs
    )
    
    if USE_MIXUP and torch.cuda.is_available():
        # Add mixup callback for better generalization
        try:
            learn.add_cb(MixUp(MIXUP_ALPHA))
            print(f"Added MixUp with alpha={MIXUP_ALPHA}")
        except Exception as e:
            print(f"Could not add MixUp callback: {e}")
            print("Continuing without MixUp")
    
    # Check FastAI version
    import fastai
    fastai_version = fastai.__version__
    print(f"FastAI version: {fastai_version}")
    
    # Try to enable mixed precision if CUDA is available
    if torch.cuda.is_available():
        try:
            learn.to_fp16()
            print("Mixed precision training enabled")
        except:
            print("Could not enable mixed precision")
    
    # Use learning rate finder to find optimal learning rate
    print("Finding optimal learning rate...")
    suggested_lr = 1e-3  # Default value in case LR finder fails
    
    try:
        lr_finder = learn.lr_find(suggest_funcs=(minimum, steep))
        if hasattr(lr_finder, 'valley') and lr_finder.valley is not None:
            suggested_lr = lr_finder.valley
            print(f"Using valley method - suggested learning rate: {suggested_lr:.3e}")
        elif hasattr(lr_finder, 'minimum') and lr_finder.minimum is not None:
            suggested_lr = lr_finder.minimum
            print(f"Using minimum method - suggested learning rate: {suggested_lr:.3e}")
        elif hasattr(lr_finder, 'steep') and lr_finder.steep is not None:
            suggested_lr = lr_finder.steep
            print(f"Using steep method - suggested learning rate: {suggested_lr:.3e}")
        else:
            print(f"No optimal learning rate found, using default: {suggested_lr:.3e}")
    except Exception as e:
        print(f"Could not find optimal learning rate: {e}")
        print(f"Using default learning rate: {suggested_lr:.3e}")
    
    # Ensure learning rate is within a reasonable range
    suggested_lr = max(min(suggested_lr, 1e-2), 1e-5)  # Clamp between 1e-5 and 1e-2
    
    # Use one-cycle policy for better training
    print("Step 1: Training the head with frozen body...")
    learn.fit_one_cycle(NUM_EPOCHS, suggested_lr)
    
    # Save intermediate model
    try:
        learn.save('stage1')
        print("Saved stage 1 model")
    except Exception as e:
        print(f"Could not save intermediate model: {e}")
    
    # Step 2: Gradual unfreezing for better fine-tuning
    print("Step a: Unfreezing the last few layers...")
    learn.unfreeze()
    
    # Find new learning rate after unfreezing
    try:
        # Use a simpler approach - take a fraction of the original learning rate
        max_lr = suggested_lr / 5
        min_lr = max_lr / 10
        print(f"Using discriminative learning rates: {min_lr:.3e} to {max_lr:.3e}")
    except Exception as e:
        print(f"Could not determine fine-tuning learning rates: {e}")
        max_lr = 1e-4  # Fallback
        min_lr = 1e-5
        print(f"Using default fine-tuning rates: {min_lr:.3e} to {max_lr:.3e}")
    
    # Step 2b: Train with discriminative learning rates
    print("Step 2: Fine-tuning the entire model with discriminative learning rates...")
    learn.fit_one_cycle(FINE_TUNE_EPOCHS, lr_max=slice(min_lr, max_lr))
    
    # Save the intermediate model after fine-tuning
    try:
        learn.save('stage2')
        print("Saved stage 2 model")
    except Exception as e:
        print(f"Could not save stage 2 model: {e}")
    
    # Step 3: If using progressive resizing, train with larger images
    if USE_PROGRESSIVE_RESIZING and dls_final is not None:
        print(f"Step 3: Fine-tuning with larger images ({FINAL_IMAGE_SIZE}x{FINAL_IMAGE_SIZE})...")
        
        # Make sure we free up GPU memory before creating a new model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared GPU cache for new model")
        
        # Transfer weights to new learner with larger images
        learn_final = vision_learner(
            dls_final,
            'resnet50',
            metrics=[error_rate, accuracy],
            loss_func=LabelSmoothingCrossEntropy(eps=LABEL_SMOOTHING),
            pretrained=False  # Don't use pretrained weights as we'll load from previous phase
        )
        
        # Make sure the model is on the correct device before loading
        learn_final.model = learn_final.model.to(device)
        print(f"Created new model on {device}")
        
        # Load weights from previous model using safe loader
        if not safe_load_model(learn_final, 'stage2'):
            print("Could not load stage2 model for final fine-tuning. Using pretrained weights instead.")
            # Create a new learner with pretrained weights instead
            learn_final = vision_learner(
                dls_final,
                'resnet50',
                metrics=[error_rate, accuracy],
                loss_func=LabelSmoothingCrossEntropy(eps=LABEL_SMOOTHING),
                pretrained=True  # Fall back to pretrained weights
            )
            # Ensure the model is on the correct device
            learn_final.model = learn_final.model.to(device)
        
        # Enable mixed precision for fine-tuning
        if torch.cuda.is_available():
            try:
                learn_final.to_fp16()
                print("Mixed precision enabled for final fine-tuning")
            except Exception as e:
                print(f"Could not enable mixed precision: {e}")
        
        # Verify all model parameters are on the same device
        try:
            model_devices = set(p.device for p in learn_final.model.parameters())
            print(f"Model parameters are on devices: {model_devices}")
            if len(model_devices) > 1:
                print("WARNING: Model parameters are on different devices!")
                # Move all to the same device
                learn_final.model = learn_final.model.to(device)
                print(f"Forced all model parameters to {device}")
        except Exception as e:
            print(f"Error checking model devices: {e}")
        
        # Create a fresh optimizer to avoid device mismatch
        learn_final.create_opt()
        
        # Final fine-tuning with lower learning rate
        final_lr = min_lr / 2
        print(f"Starting final fine-tuning with lr={final_lr}")
        
        try:
            learn_final.fit_one_cycle(5, lr_max=final_lr)
            print("Final fine-tuning completed successfully")
        except Exception as e:
            print(f"Error during final fine-tuning: {e}")
            print("Continuing with previous model")
            learn_final = learn
        
        # Save final model
        try:
            learn_final.save('final_model')
            print("Saved final model with larger resolution")
        except Exception as e:
            print(f"Could not save final model: {e}")
        
        # Use the final model for further steps
        learn = learn_final
    
    # Create model directory if needed
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Save the model
    try:
        # First approach: Save method - convert Path to string to avoid nesting issues
        model_dir = str(MODEL_PATH)
        if not model_dir.endswith('/'):
            model_dir += '/'
        model_path = f"{model_dir}final_model"
        learn.save(model_path)
        print(f"Model saved to {model_path}.pth")
    except Exception as e1:
        print(f"Could not save model with save() method: {e1}")
        
        # Approach 2: Save to current directory and then copy
        try:
            # Save to current directory
            learn.save('final_model')
            print("Model saved to final_model.pth")
            
            # Manually copy the file to models directory
            import shutil
            curr_dir = os.getcwd()
            source_file = os.path.join(curr_dir, 'final_model.pth')
            dest_file = os.path.join(model_dir, 'final_model.pth')
            
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
                print(f"Copied model from {source_file} to {dest_file}")
            else:
                print(f"Source model file {source_file} not found")
                
            # Try to use export method as a last resort
            try:
                export_path = os.path.join(model_dir, MODEL_NAME)
                learn.export(export_path)
                print(f"Model exported to {export_path}")
            except Exception as e3:
                print(f"Could not export model: {e3}")
        except Exception as e2:
            print(f"All saving methods failed: {e2}")
    
    return learn

def predict_celebrity(model, image_path):
    """Predict the celebrity identity for a given image with confidence calibration."""
    # Load and process the image
    img = PILImage.create(image_path)
    
    # Get prediction and probabilities
    prediction, _, probs = model.predict(img)
    raw_confidence = float(torch.max(probs))
    
    # Apply temperature scaling for better calibration (reduces overconfidence)
    temperature = 1.5  # Higher temperature values produce softer probability distributions
    calibrated_probs = torch.nn.functional.softmax(torch.log(probs) / temperature, dim=0)
    calibrated_confidence = float(torch.max(calibrated_probs))
    
    # Get top-3 predictions with confidences for better analysis
    top_indices = torch.topk(calibrated_probs, k=min(3, len(calibrated_probs))).indices
    top_classes = [model.dls.vocab[i.item()] for i in top_indices]
    top_confidences = [float(calibrated_probs[i]) for i in top_indices]
    
    # Pack the results
    top_predictions = list(zip(top_classes, top_confidences))
    
    return prediction, calibrated_confidence, top_predictions

def main():
    """Main function to run the celebrity recognition pipeline."""
    print("Celebrity Recognition Application")
    print("---------------------------------")
    
    # Load identity data
    identity_df = load_identity_data()
    
    # Try to load a saved model
    learn = None
    model_loaded = False
    
    # Check different possible model files (ensure all are Path objects)
    possible_model_files = [
        MODEL_PATH/MODEL_NAME,           # Default .pkl file
        MODEL_PATH/'model.pth',          # .pth file
        MODEL_PATH/'final_model.pth',    # Final model in models directory
        Path('final_model.pth'),         # Another possible location
        Path('stage2.pth'),              # Second stage model
        Path('stage1.pth'),              # Intermediate model
    ]
    
    # Try to load any available model
    for model_file in possible_model_files:
        # Ensure model_file is a Path object
        model_file = Path(model_file)
        
        if model_file.exists():
            print(f"Attempting to load model from {model_file}")
            try:
                if str(model_file).endswith('.pkl'):
                    # Use load_learner for .pkl files
                    learn = load_learner(model_file)
                else:
                    # For .pth files, we need to create a learner first, then load the model
                    # First create empty dataloaders
                    # Use a larger image size for better prediction results
                    tempdf = identity_df.head(10)  # Just use a small subset for creating the structure
                    temp_dls = setup_data_loaders(tempdf, img_size=FINAL_IMAGE_SIZE)
                    
                    # Create a learner with the same architecture
                    learn = vision_learner(
                        temp_dls,
                        'resnet50',
                        metrics=[error_rate, accuracy],
                        loss_func=LabelSmoothingCrossEntropy(eps=LABEL_SMOOTHING),
                        normalize=True
                    )
                    
                    # Load the saved weights - strip the .pth extension
                    model_name = str(model_file).replace('.pth', '')
                    # Use the safe loader
                    if safe_load_model(learn, model_name):
                        print(f"Successfully loaded model from {model_file}")
                        model_loaded = True
                        break
                    else:
                        continue
                
                print(f"Successfully loaded model from {model_file}")
                model_loaded = True
                break
            except Exception as e:
                print(f"Failed to load model from {model_file}: {e}")
                continue
    
    # If no model was loaded, train a new one
    if not model_loaded:
        print("No existing model could be loaded. Training new model...")
        
        # Check for CUDA and manage GPU memory
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            # Try to set memory allocation mode to reduce fragmentation
            try:
                # Set environment variable for memory allocation
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                print("Enabled expandable memory segments for CUDA")
            except:
                print("Could not set CUDA memory allocation configuration")
        else:
            print("CUDA not available. Using CPU for training (will be slower)")
        
        # Setup data loaders for initial training
        dataloaders = setup_data_loaders(identity_df, img_size=IMAGE_SIZE)
        
        # Setup data loaders for final fine-tuning with larger images if using progressive resizing
        dataloaders_final = None
        if USE_PROGRESSIVE_RESIZING:
            print(f"Setting up data loaders for final fine-tuning with image size {FINAL_IMAGE_SIZE}...")
            dataloaders_final = setup_data_loaders(identity_df, img_size=FINAL_IMAGE_SIZE)
        
        # Train model
        try:
            learn = train_model(dataloaders, dataloaders_final)
            print("\nTraining completed successfully!")
            
            # Try to create model interpretation safely
            try:
                print("\nGenerating confusion matrix...")
                # Create results directory first
                os.makedirs(RESULTS_DIR, exist_ok=True)
                
                # Get predictions manually with better error handling
                try:
                    # Check if we have a valid learner with a valid DataLoaders object
                    if learn is None or learn.dls is None:
                        print("Warning: No valid model or dataloaders found, cannot create confusion matrix")
                    else:
                        # Use a smaller validation set to get predictions reliably
                        preds = []
                        targets = []
                        
                        # Use a try block specifically for get_preds
                        try:
                            # Try to get predictions in small batches to avoid memory issues
                            dl = learn.dls.valid
                            if dl.n == 0:
                                print("Warning: Validation set is empty, cannot create confusion matrix")
                            else:
                                # Get predictions in batches
                                for b in range(min(5, len(dl))):  # Use at most 5 batches for the matrix
                                    batch = dl.one_batch()
                                    batch_preds, batch_targets = learn.get_preds(dl=[batch])
                                    preds.append(batch_preds)
                                    targets.append(batch_targets)
                                
                                # Combine all predictions
                                if preds and targets:
                                    preds = torch.cat(preds)
                                    targets = torch.cat(targets)
                                    
                                    # Verify we got actual predictions
                                    if len(preds) == 0 or len(targets) == 0:
                                        print("Warning: No predictions or targets found, cannot create confusion matrix")
                                    else:
                                        # Convert to numpy arrays
                                        preds_np = preds.argmax(dim=1).numpy()
                                        targets_np = targets.numpy()
                                        
                                        # Get class names
                                        class_names = learn.dls.vocab
                                        
                                        # Create and save confusion matrix directly
                                        from sklearn.metrics import confusion_matrix
                                        import seaborn as sns
                                        
                                        # Get top 15 classes by frequency
                                        unique_targets, counts = np.unique(targets_np, return_counts=True)
                                        top_indices = np.argsort(-counts)[:min(15, len(unique_targets))]  # Top 15 most frequent classes
                                        
                                        # Create confusion matrix for these classes
                                        top_class_names = [class_names[i] for i in top_indices]
                                        
                                        # Create a mask for the top classes
                                        mask_preds = np.isin(preds_np, top_indices)
                                        mask_targets = np.isin(targets_np, top_indices)
                                        mask = mask_preds & mask_targets
                                        
                                        # Filter predictions and targets
                                        filtered_preds = preds_np[mask]
                                        filtered_targets = targets_np[mask]
                                        
                                        # Create confusion matrix
                                        cm = confusion_matrix(filtered_targets, filtered_preds)
                                        
                                        # Plot
                                        plt.figure(figsize=(12, 12))
                                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                                  xticklabels=top_class_names,
                                                  yticklabels=top_class_names)
                                        plt.title('Confusion Matrix (Top 15 Classes)')
                                        plt.ylabel('True Label')
                                        plt.xlabel('Predicted Label')
                                        plt.xticks(rotation=45, ha='right')
                                        
                                        # Save figure to results directory
                                        plt.savefig(RESULTS_DIR/'confusion_matrix.png')
                                        print(f"Confusion matrix saved to {RESULTS_DIR/'confusion_matrix.png'}")
                                        plt.close()
                                else:
                                    print("Warning: Could not generate any predictions for confusion matrix")
                        except Exception as e:
                            print(f"Error generating predictions batches: {e}")
                except Exception as e:
                    print(f"Error in matrix generation: {e}")
            except Exception as e:
                print(f"Could not create confusion matrix: {e}")
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("\nGPU ran out of memory. Try with a smaller batch size or image size.")
                print("You can use train_lowres.py instead which is optimized for lower resources.")
            else:
                # Re-raise if it's not a memory error
                raise
    
    # Verify the model is loaded properly
    if learn is None:
        print("Fatal error: Could not load or train a model.")
        return
        
    # Interactive prediction mode
    while True:
        image_path = input("\nEnter the path to an image (or 'q' to quit): ")
        if image_path.lower() == 'q':
            break
        
        try:
            celebrity, confidence, top_predictions = predict_celebrity(learn, image_path)
            print(f"Predicted celebrity: {celebrity}")
            print(f"Confidence: {confidence:.2%}")
            print("\nTop predictions:")
            for idx, (pred, conf) in enumerate(top_predictions, 1):
                print(f"{idx}. {pred}: {conf:.2%}")
        except Exception as e:
            print(f"Error processing image: {e}")
            print("Make sure the path is correct and the image exists.")

if __name__ == "__main__":
    main() 