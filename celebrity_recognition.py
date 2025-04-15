#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastai.vision.all import *
from fastai.callback.schedule import minimum, steep  # Add missing imports for LR finder
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import torch.serialization
import sys
import time
import logging
import traceback
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

# Focal Loss implementation for dealing with class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = float(alpha)  # Ensure float type
        self.gamma = float(gamma)  # Ensure float type
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Test Time Augmentation helper function
def tta_inference(model, img_tensor, num_augments=5):
    """Perform test-time augmentation and average predictions"""
    model.eval()
    # Original prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
    # Apply different augmentations and average predictions
    tta_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.1, p=1.0)
    ]
    
    for t in tta_transforms:
        # Apply transform
        aug_tensor = t(img_tensor)
        
        # Predict on augmented image
        with torch.no_grad():
            aug_output = model(aug_tensor)
            aug_probs = torch.nn.functional.softmax(aug_output, dim=1)
            
        # Accumulate probabilities
        probs += aug_probs
    
    # Average probabilities
    probs = probs / (len(tta_transforms) + 1)
    
    return probs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)
logging.getLogger('PIL').setLevel(logging.WARNING)  # Reduce PIL logging noise

# Configuration
DATA_DIR = Path("/home/dave/torch/celeba_images/img_align_celeba")
IDENTITY_FILE = "list_identity_celeba.txt"
MODEL_PATH = Path("models")
RESULTS_DIR = Path("evaluation_results")
MODEL_NAME = "celebrity_recognition_model.pkl"
BATCH_SIZE = 8  # Reduced to handle smaller datasets
IMAGE_SIZE = 224  # Standard size for pretrained models
FINAL_IMAGE_SIZE = 320  # Larger final size for better detail
NUM_EPOCHS = 10  # More epochs for smaller dataset
FINE_TUNE_EPOCHS = 15  # Fine-tuning epochs
LEARNING_RATE = 5e-4  # Slightly higher for smaller dataset
MIN_IMAGES_PER_CELEB = 10  # Reduced minimum to work with smaller datasets
MAX_CELEBRITIES = 10  # Focus on fewer celebrities for better accuracy
USE_MIXUP = True
MIXUP_ALPHA = 0.1  # Lower for less aggressive mixing
LABEL_SMOOTHING = 0.03  # Reduced for more confident predictions
USE_PROGRESSIVE_RESIZING = True
ARCHITECTURE = 'resnet50'  # Use more stable architecture for smaller dataset
USE_PURE_PYTORCH = True  # Use PyTorch implementation
VALIDATION_PCT = 0.2  # Increased validation percentage
WEIGHT_DECAY = 1e-2  # Reduced weight decay for smaller dataset
USE_TEST_TIME_AUGMENTATION = True  # Add test-time augmentation
USE_GRADIENT_ACCUMULATION = True  # Add gradient accumulation
GRAD_ACCUM_STEPS = 2  # Accumulate gradients over 2 steps
USE_FOCAL_LOSS = True  # Use focal loss for class imbalance
USE_COSINE_ANNEALING = True  # Use cosine annealing scheduler

# Set device globally to ensure consistency
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define functions at the module level to avoid pickling issues
def get_x(r): return DATA_DIR/r['image_id']
def get_y(r): return r['identity_name']

def log_info(msg):
    """Log information with timestamp"""
    logging.info(msg)
    print(msg)

def log_memory():
    """Log memory usage if GPU is available"""
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    cached = torch.cuda.memory_reserved() / (1024 * 1024)
    log_info(f"GPU Memory: {allocated:.2f}MB allocated, {cached:.2f}MB cached")

# PyTorch Direct Dataset Implementation
class DirectImageDataset(Dataset):
    """Dataset that loads images directly from disk paths"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image directly using PIL
        try:
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                img = self.transform(img)
                
            # Get corresponding label
            label = self.labels[idx]
            
            return img, label
        except Exception as e:
            log_info(f"Error loading image {idx}: {e}")
            # Create a black image as fallback
            if self.transform:
                img = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            else:
                img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
                img = transforms.ToTensor()(img)
            return img, self.labels[idx]

def create_pure_pytorch_datasets(df, img_size=IMAGE_SIZE):
    """Create PyTorch datasets from DataFrame without FastAI dependencies"""
    log_info("Creating pure PyTorch datasets...")
    
    # Split data into train and validation sets (stratified)
    train_df, valid_df = train_test_split(
        df, test_size=VALIDATION_PCT, stratify=df['identity_name'], random_state=42
    )
    
    # Reset indices for cleaner dataframes
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
    log_info(f"Training set: {len(train_df)} images")
    log_info(f"Validation set: {len(valid_df)} images")
    
    # Get class vocabulary mapping
    vocab = sorted(df['identity_name'].unique())
    label_to_idx = {label: i for i, label in enumerate(vocab)}
    
    # Create label indices
    train_df['label_idx'] = train_df['identity_name'].apply(lambda x: label_to_idx[x])
    valid_df['label_idx'] = valid_df['identity_name'].apply(lambda x: label_to_idx[x])
    
    # Enhanced train transforms with more augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((img_size+32, img_size+32)),  # Resize larger then crop for better scale diversity
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
        transforms.RandomGrayscale(p=0.01),
        # Conditionally add RandomAutocontrast based on availability
        *([transforms.RandomAutocontrast(p=0.2)] if hasattr(transforms, 'RandomAutocontrast') else []),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Simpler transforms for validation
    valid_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DirectImageDataset(
        train_df['image_path'].tolist(),
        train_df['label_idx'].tolist(),
        transform=train_transforms
    )
    
    valid_dataset = DirectImageDataset(
        valid_df['image_path'].tolist(),
        valid_df['label_idx'].tolist(),
        transform=valid_transforms
    )
    
    log_info(f"Created datasets with {len(train_dataset)} training and {len(valid_dataset)} validation samples")
    
    # Create dataloaders with optional pin memory
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Drop last batch to avoid size mismatches
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create class weights for loss function weighting
    class_counts = df['identity_name'].value_counts()
    # Convert to class indices order
    class_weights = torch.zeros(len(vocab))
    total_samples = float(len(df))
    
    for label, idx in label_to_idx.items():
        count = class_counts[label]
        # Improved inverse frequency weighting with smoothing
        class_weights[idx] = (total_samples / (len(vocab) * count)) ** 0.5  # Square root to reduce extreme values
    
    # Normalize weights
    class_weights = class_weights / class_weights.mean()
    log_info(f"Created balanced class weights with range: {class_weights.min():.2f} - {class_weights.max():.2f}")
    
    # Create a class similar to FastAI's DataLoaders for compatibility
    class PyTorchDataLoaders:
        def __init__(self, train_dl, valid_dl, vocab, class_weights=None):
            self.train = train_dl
            self.valid = valid_dl
            self.vocab = vocab
            self.c = len(vocab)  # Number of classes
            self.class_weights = class_weights
            
        def one_batch(self):
            return next(iter(self.train))
            
        def show_batch(self, max_n=9):
            """Display a batch of images"""
            batch = self.one_batch()
            imgs, labels = batch
            
            # Convert to numpy for display
            imgs = imgs.cpu().numpy()
            
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            imgs = imgs * std + mean
            
            # Clip to valid range
            imgs = np.clip(imgs, 0, 1)
            
            # Display images
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            for i, ax in enumerate(axes.flat):
                if i < len(imgs):
                    # Convert to HWC format for display
                    img = np.transpose(imgs[i], (1, 2, 0))
                    ax.imshow(img)
                    label_idx = labels[i].item()
                    label_name = self.vocab[label_idx]
                    ax.set_title(f"Class: {label_name}")
                    ax.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    return PyTorchDataLoaders(train_loader, valid_loader, vocab, class_weights)

def check_image_directory(directory):
    """Check if the image directory is populated and extract sample files."""
    try:
        if not os.path.exists(directory):
            log_info(f"ERROR: Directory {directory} does not exist!")
            return False, []
            
        files = list(Path(directory).glob("*.jpg"))
        sample_files = files[:5] if files else []
        
        file_count = len(files)
        log_info(f"Found {file_count} files in {directory}")
        
        if file_count == 0:
            log_info(f"ERROR: No image files found in {directory}")
            
            # Check if there's a zip file that might need extraction
            zip_file = Path(str(directory) + '.zip')
            if zip_file.exists():
                log_info(f"Found zip file {zip_file}, you may need to extract it")
            
            return False, sample_files
            
        log_info(f"Sample files: {[f.name for f in sample_files]}")
        return True, sample_files
    
    except Exception as e:
        log_info(f"Error checking image directory: {e}")
        return False, []

def load_identity_data():
    """Load celebrity identity data from the text file with improved filtering."""
    log_info("Loading identity data...")
    
    # Check if identity file exists
    if not os.path.exists(IDENTITY_FILE):
        log_info(f"ERROR: Identity file not found at {os.path.abspath(IDENTITY_FILE)}")
        raise FileNotFoundError(f"Identity file {IDENTITY_FILE} not found")
    
    # Check if data directory is properly populated
    has_images, sample_files = check_image_directory(DATA_DIR)
    if not has_images:
        log_info("WARNING: Image directory appears to be empty or has issues")
        
    # Skip the first few lines which contain metadata (count + header)
    try:
        df = pd.read_csv(IDENTITY_FILE, sep=r"\s+", skiprows=2, 
                         names=["image_id", "identity_name"])
        log_info(f"Loaded {len(df)} image identity records")
    except Exception as e:
        log_info(f"ERROR reading identity file: {e}")
        try:
            # Try without skipping rows
            df = pd.read_csv(IDENTITY_FILE, sep=r"\s+", names=["image_id", "identity_name"])
            log_info(f"Successfully loaded {len(df)} records without skiprows")
        except Exception as e2:
            log_info(f"Failed to read identity file: {e2}")
            raise
    
    # Add image paths and filter for existing images
    df['image_path'] = df['image_id'].apply(lambda x: DATA_DIR/x)
    before_count = len(df)
    
    # Filter for existing images
    df = df[df['image_path'].apply(lambda x: x.exists())]
    if before_count > len(df):
        log_info(f"Filtered out {before_count - len(df)} missing images")
    
    if len(df) == 0:
        log_info("ERROR: No valid images found! Check paths and file structure.")
        
        # Check for zip file that might need extraction
        zip_file = Path("img_align_celeba.zip")
        if zip_file.exists():
            log_info(f"Found {zip_file} which needs to be extracted.")
            log_info("Please run the following command to extract images:")
            log_info("unzip img_align_celeba.zip")
            raise RuntimeError("Please extract the image dataset before continuing. See instructions above.")
        
        # Try to be more permissive by reducing threshold
        global MIN_IMAGES_PER_CELEB
        MIN_IMAGES_PER_CELEB = 5
        log_info(f"Reduced MIN_IMAGES_PER_CELEB to {MIN_IMAGES_PER_CELEB}")
        
        # Create dummy data for testing if needed
        log_info("Creating dummy data for debugging purposes")
        dummy_data = {"image_id": [], "identity_name": [], "image_path": []}
        # Create 50 dummy entries for testing
        for i in range(50):
            dummy_data["image_id"].append(f"img_{i}.jpg")
            dummy_data["identity_name"].append(f"celeb_{i//10}")
            dummy_data["image_path"].append(DATA_DIR/f"img_{i}.jpg")
            
        df = pd.DataFrame(dummy_data)
        log_info(f"Created dummy DataFrame with {len(df)} entries and {df['identity_name'].nunique()} celebrities")
        
    # Remove celebrities with too few images
    celeb_counts = df['identity_name'].value_counts()
    log_info(f"Celebrity counts: {celeb_counts}")
    
    valid_celebs = celeb_counts[celeb_counts >= MIN_IMAGES_PER_CELEB].index
    if len(valid_celebs) == 0:
        log_info(f"WARNING: No celebrities with at least {MIN_IMAGES_PER_CELEB} images")
        # Use all celebrities as a fallback
        valid_celebs = celeb_counts.index
        log_info(f"Using all {len(valid_celebs)} celebrities regardless of image count")
    
    df = df[df['identity_name'].isin(valid_celebs)]
    log_info(f"After minimum image filtering: {len(df)} images of {len(valid_celebs)} celebrities")
    
    # Limit to top MAX_CELEBRITIES with most images for faster training
    if MAX_CELEBRITIES > 0 and len(valid_celebs) > MAX_CELEBRITIES:
        top_celebs = df['identity_name'].value_counts().head(MAX_CELEBRITIES).index
        df = df[df['identity_name'].isin(top_celebs)]
        log_info(f"Limited to top {MAX_CELEBRITIES} celebrities ({len(df)} images)")
    
    # Calculate and print class distribution stats
    if len(df) > 0:
        class_counts = df['identity_name'].value_counts()
        min_images = class_counts.min()
        max_images = class_counts.max()
        mean_images = class_counts.mean()
        log_info(f"Class distribution - Min: {min_images}, Max: {max_images}, Avg: {mean_images:.1f} images per celebrity")
    else:
        log_info("ERROR: DataFrame is empty after filtering!")
        raise ValueError("No valid data after filtering. Check paths and requirements.")
    
    return df

def setup_data_loaders(df, img_size=IMAGE_SIZE):
    """Set up data loaders - either FastAI or PyTorch based on configuration."""
    # Make sure model directory exists
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Use PyTorch implementation if specified
    if USE_PURE_PYTORCH:
        return create_pure_pytorch_datasets(df, img_size)
    
    # Otherwise use the FastAI implementation (original code)
    # Split data into train and validation sets (stratified)
    train_df, valid_df = train_test_split(
        df, test_size=0.15, stratify=df['identity_name'], random_state=42
    )
    
    # Reset indices for cleaner dataframes
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
    log_info(f"Training set: {len(train_df)} images")
    log_info(f"Validation set: {len(valid_df)} images")
    
    # Create class weights for handling imbalance
    unique_labels = df['identity_name'].unique()
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=df['identity_name']
    )
    class_weight_dict = dict(zip(unique_labels, class_weights))
    log_info(f"Applied class weighting to handle imbalanced data")
    
    # Enhanced data augmentation for improved generalization
    # More aggressive augmentation to improve model robustness
    tfms = [
        *aug_transforms(
            do_flip=True, 
            flip_vert=False, 
            max_rotate=15.0,  # Less rotation for face recognition
            max_zoom=1.15,    # Moderate zoom for faces
            max_lighting=0.3, # Lighting variation
            max_warp=0.1,     # Less warping for faces
            p_affine=0.75,
            p_lighting=0.75,
        ),
        Normalize.from_stats(*imagenet_stats)
    ]
    
    # Add additional augmentation for more robustness
    if USE_MIXUP:
        # Use RandomErasing with compatible parameters
        tfms.append(RandomErasing(p=0.5, max_count=3))
    
    # Create FastAI DataBlock with more focused augmentation for faces
    celeb_data = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=get_x,
        get_y=get_y,
        splitter=ColSplitter(),
        item_tfms=[
            Resize(img_size, method=ResizeMethod.Squish),
            # Center crop focuses more on the face
            RandomResizedCrop(img_size, min_scale=0.9, ratio=(0.95, 1.05))
        ],
        batch_tfms=tfms
    )
    
    # Add a column for splitting the data
    df['is_valid'] = np.where(df['image_id'].isin(valid_df['image_id']), True, False)
    
    # Create dataloaders with smaller batch size for better gradient updates
    dls = celeb_data.dataloaders(df, bs=BATCH_SIZE, num_workers=4)
    
    # Store class weights for use in loss function
    dls.class_weights = torch.FloatTensor([class_weight_dict[c] for c in dls.vocab])
    
    return dls

# Helper function to safely load models
def safe_load_model(learner, name):
    """Safely load a model with PyTorch 2.6+ compatibility."""
    try:
        # First try the normal loading
        learner.load(name)
        log_info(f"Successfully loaded model {name} with normal loading")
    except Exception as e1:
        log_info(f"Error loading model with normal method: {e1}")
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
            log_info(f"Attempting to load with weights_only=False from {file} to device {device}")
            
            # Add safe globals for PyTorch 2.6+
            torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
            state = torch.load(file, map_location=torch.device(device), weights_only=False)
            
            # Apply the state to the model
            learner.model.load_state_dict(state['model'], strict=False)
            log_info(f"Successfully loaded model weights for {name}")
            
            # Explicitly move the model to the correct device
            learner.model.to(device)
            log_info(f"Moved model to {device}")
            
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
                    log_info("Loaded and aligned optimizer state")
                except Exception as e:
                    log_info(f"Could not load optimizer state: {e}")
                    log_info("Continuing with new optimizer")
                    # Create a fresh optimizer
                    learner.create_opt()
            else:
                # Create a fresh optimizer to avoid device mismatch issues
                learner.create_opt()
                log_info("Created new optimizer")
                    
            return True
                    
        except Exception as e2:
            log_info(f"Error loading model with weights_only=False: {e2}")
            try:
                # Last resort: manual loading of just the model weights
                file = f"{name}.pth"
                if not Path(file).exists():
                    # Try models directory
                    file = Path('models')/f"{name}.pth"
                    
                # Use a custom unpickling approach
                log_info(f"Attempting manual load from {file}")
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
                    log_info(f"Successfully loaded model weights manually for {name} to {device}")
                    # Create fresh optimizer
                    learner.create_opt()
                    return True
                else:
                    log_info("Could not find model weights in saved state")
                    return False
                    
            except Exception as e3:
                log_info(f"All loading methods failed: {e3}")
                return False
    return True

def is_arch_available(arch_name):
    """Check if an architecture is available"""
    try:
        # Try to create a learner with this architecture
        temp_data = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=lambda x: x,  # Dummy function
            get_y=lambda x: 0    # Dummy function
        ).dataloaders([1], bs=1)
        
        learn = vision_learner(temp_data, arch_name)
        return True
    except Exception as e:
        log_info(f"Architecture {arch_name} is not available: {e}")
        return False

def create_vision_learner(dls, arch_name=ARCHITECTURE, metrics=None, **kwargs):
    """Create a vision learner with fallback options if the specified architecture isn't available"""
    if metrics is None:
        metrics = [accuracy]
        
    # Check if the requested architecture is available
    if not is_arch_available(arch_name):
        log_info(f"Requested architecture {arch_name} not available, falling back to resnet50")
        arch_name = 'resnet50'
        
    # Add label smoothing for more robust learning
    if LABEL_SMOOTHING > 0:
        log_info(f"Using label smoothing with alpha={LABEL_SMOOTHING}")
        loss_func = LabelSmoothingCrossEntropy(eps=LABEL_SMOOTHING)
    else:
        loss_func = None
        
    # Create the learner with proper device placement
    learn = vision_learner(dls, arch_name, metrics=metrics, loss_func=loss_func, **kwargs)
    
    # Ensure model is on the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learn.model = learn.model.to(device)
    
    return learn

# New PyTorch model creation function
def create_pytorch_model(arch_name=ARCHITECTURE, num_classes=None):
    """Create a PyTorch model with the specified architecture"""
    log_info(f"Creating PyTorch model with {arch_name} architecture")
    
    # Create model based on architecture name
    if arch_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Add dropout and feature normalization between the pool and the fc layer
        num_features = model.fc.in_features
        
        # Replace the final layer with a sequence that includes dropout
        model.fc = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    elif arch_name == 'convnext_small':
        try:
            model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
            # Get feature dimension
            num_features = model.classifier[-1].in_features
            
            # Replace the final layers
            model.classifier = nn.Sequential(
                # Keep the first part (LayerNorm)
                model.classifier[0],
                # Replace the linear layer with a sequence
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        except Exception as e:
            log_info(f"ConvNext not available: {e}, falling back to ResNet50")
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            # Add dropout and feature normalization
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
    else:
        # Default fallback to ResNet50
        log_info(f"Architecture {arch_name} not recognized, using ResNet50")
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Add dropout and feature normalization
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
    
    # Initialize the new layers properly (this can help with convergence)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    # Move model to the correct device
    model = model.to(DEVICE)
    log_info(f"Model created and moved to {DEVICE}")
    log_info(f"Added dropout and normalization layers for better regularization")
    
    return model

# Load a PyTorch model from file
def load_pytorch_model(model, model_path):
    """Load saved PyTorch model weights"""
    try:
        log_info(f"Loading model from {model_path}")
        # Check if file exists
        if not os.path.exists(model_path):
            log_info(f"Model file {model_path} not found")
            return False
            
        # Load state dict
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # If it's a dictionary with 'model' key (like FastAI saves)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            model.load_state_dict(state_dict['model'], strict=False)
        else:
            # Direct state dict
            model.load_state_dict(state_dict, strict=False)
            
        log_info("Model loaded successfully")
        return True
    except Exception as e:
        log_info(f"Error loading model: {e}")
        return False

# Pure PyTorch training loop
def train_pytorch_model(model, dls, epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """Train a model using pure PyTorch"""
    log_info(f"Training PyTorch model for {epochs} epochs with learning rate {lr}")
    log_memory()
    
    # Define loss function with class weights if available
    if hasattr(dls, 'class_weights') and dls.class_weights is not None:
        weights = dls.class_weights.to(DEVICE)
        log_info("Using weighted loss function")
        
        if USE_FOCAL_LOSS:
            log_info("Using Focal Loss with class weights")
            criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    else:
        log_info("Using standard loss function")
        if USE_FOCAL_LOSS:
            log_info("Using Focal Loss")
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # Define optimizer with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    if USE_COSINE_ANNEALING:
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=epochs, 
            T_mult=1, 
            eta_min=lr/20
        )
        log_info("Using Cosine Annealing scheduler with warm restarts")
    else:
        # Reduce on plateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=lr/20
        )
        log_info("Using ReduceLROnPlateau scheduler")
    
    # Training loop
    start_time = time.time()
    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_PATH, 'best_model.pth')
    
    # Track metrics
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_top3_acc': []
    }
    
    # Early stopping parameters
    patience = 5
    early_stop_counter = 0
    
    # Gradient accumulation setup
    grad_accum_steps = GRAD_ACCUM_STEPS if USE_GRADIENT_ACCUMULATION else 1
    effective_batch_size = BATCH_SIZE * grad_accum_steps
    log_info(f"Using gradient accumulation with {grad_accum_steps} steps (effective batch size: {effective_batch_size})")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        log_info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Reset gradients for first accumulation step
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(dls.train):
            # Move data to device
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Check and fix tensor dimensions if needed
            if len(inputs.shape) != 4:
                log_info(f"Fixing input shape: {inputs.shape} -> 4D tensor")
                if len(inputs.shape) == 5:  # Case: [batch, c, c, h, w]
                    if inputs.shape[1] == 3 and inputs.shape[2] == 3:
                        inputs = inputs[:, :, 0, :, :]
                    else:
                        # Reshape to 4D tensor
                        inputs = inputs.reshape(-1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1])
                elif len(inputs.shape) == 3:  # Missing batch dimension
                    inputs = inputs.unsqueeze(0)
            
            # Implement Mixup data augmentation if enabled
            if USE_MIXUP and epoch < epochs - 2:  # No mixup in final epochs
                # Generate mixing parameter
                alpha = MIXUP_ALPHA
                lam = np.random.beta(alpha, alpha)
                
                # Create mixed batch
                batch_size = inputs.size(0)
                index = torch.randperm(batch_size).to(DEVICE)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                targets_a, targets_b = targets, targets[index]
                
                # Forward pass on mixed inputs
                outputs = model(mixed_inputs)
                
                # Compute loss with mixup targets
                loss_a = criterion(outputs, targets_a)
                loss_b = criterion(outputs, targets_b)
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                # Standard forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Optimizer step after accumulating gradients
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dls.train):
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Reset gradients
                optimizer.zero_grad()
            
            # Track statistics (use full loss value for logging)
            train_loss += (loss.item() * grad_accum_steps) * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            
            # For mixup, count predictions based on dominant class
            if USE_MIXUP and epoch < epochs - 2:
                correct_a = (predicted == targets_a).float()
                correct_b = (predicted == targets_b).float()
                train_correct += (lam * correct_a + (1 - lam) * correct_b).sum().item()
            else:
                train_correct += (predicted == targets).sum().item()
            
            # Log progress
            if (i+1) % 20 == 0:
                log_info(f"Batch {i+1}/{len(dls.train)} - Loss: {loss.item()*grad_accum_steps:.4f}")
                log_memory()
        
        # Calculate epoch statistics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Update cosine annealing scheduler after each epoch
        if USE_COSINE_ANNEALING:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            log_info(f"Current learning rate: {current_lr:.7f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Track top-3 accuracy
        top3_correct = 0
        
        with torch.no_grad():
            for inputs, targets in dls.valid:
                # Move data to device
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                
                # Check and fix tensor dimensions if needed
                if len(inputs.shape) != 4:
                    if len(inputs.shape) == 5:  # Case: [batch, c, c, h, w]
                        if inputs.shape[1] == 3 and inputs.shape[2] == 3:
                            inputs = inputs[:, :, 0, :, :]
                        else:
                            # Reshape to 4D tensor
                            inputs = inputs.reshape(-1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1])
                    elif len(inputs.shape) == 3:  # Missing batch dimension
                        inputs = inputs.unsqueeze(0)
                
                # Forward pass - with TTA in later epochs if enabled
                if USE_TEST_TIME_AUGMENTATION and epoch >= epochs // 2:
                    # Use test-time augmentation for better predictions
                    probs = tta_inference(model, inputs)
                    outputs = torch.log(probs + 1e-8)  # Convert back to logits
                else:
                    # Standard forward pass
                    outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, targets)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                # Calculate top-3 accuracy
                _, top3_preds = torch.topk(outputs, 3, dim=1)
                for i, target in enumerate(targets):
                    if target in top3_preds[i]:
                        top3_correct += 1
        
        # Calculate validation statistics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        top3_acc = top3_correct / val_total
        
        # Update learning rate with ReduceLROnPlateau
        if not USE_COSINE_ANNEALING:
            scheduler.step(val_loss)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top3_acc'].append(top3_acc)
        
        # Log epoch results
        log_info(f"Epoch {epoch+1} results:")
        log_info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        log_info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Top-3 Acc: {top3_acc:.4f}")
        log_info(f"Epoch time: {time.time() - epoch_start:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            log_info(f"Saved new best model with accuracy {val_acc:.4f}")
            # Reset early stopping counter
            early_stop_counter = 0
        else:
            # Increment early stopping counter
            early_stop_counter += 1
            log_info(f"Validation accuracy did not improve. Early stopping counter: {early_stop_counter}/{patience}")
        
        # Save latest model
        latest_path = os.path.join(MODEL_PATH, f'latest_model_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), latest_path)
        log_info(f"Saved latest model for epoch {epoch+1}")
        
        # Plot training curves if matplotlib is available
        try:
            # Create performance plots
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title('Loss Curves')
                
                plt.subplot(1, 2, 2)
                plt.plot(history['train_acc'], label='Train Acc')
                plt.plot(history['val_acc'], label='Val Acc')
                plt.plot(history['val_top3_acc'], label='Val Top-3 Acc')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.title('Accuracy Curves')
                
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, f'training_curves_epoch{epoch+1}.png'))
                plt.close()
                log_info(f"Saved training curves to {RESULTS_DIR}/training_curves_epoch{epoch+1}.png")
        except Exception as e:
            log_info(f"Could not create training curves: {e}")
        
        # Check for early stopping
        if early_stop_counter >= patience:
            log_info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training complete
    log_info(f"Training completed in {time.time() - start_time:.2f}s")
    log_info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    return model, best_val_acc

def train_model(dls, dls_final=None):
    """Train the model using either FastAI or PyTorch implementation"""
    if USE_PURE_PYTORCH:
        # Use the pure PyTorch implementation
        log_info("Using pure PyTorch implementation for training")
        
        # Create the model
        model = create_pytorch_model(ARCHITECTURE, num_classes=len(dls.vocab))
        
        # Train the model
        trained_model, best_acc = train_pytorch_model(
            model, 
            dls, 
            epochs=NUM_EPOCHS, 
            lr=LEARNING_RATE
        )
        
        # If we're using progressive resizing and have a final size dataloader
        if USE_PROGRESSIVE_RESIZING and dls_final is not None:
            log_info(f"Fine-tuning with larger image size {FINAL_IMAGE_SIZE}")
            # Fine-tune with lower learning rate
            trained_model, final_acc = train_pytorch_model(
                trained_model,
                dls_final,
                epochs=FINE_TUNE_EPOCHS,
                lr=LEARNING_RATE/10
            )
            
        # Save the final model
        final_path = os.path.join(MODEL_PATH, MODEL_NAME.replace('.pkl', '.pth'))
        torch.save(trained_model.state_dict(), final_path)
        log_info(f"Saved final model to {final_path}")
        
        # Create a simple class to mimic the FastAI learner interface
        class PyTorchLearner:
            def __init__(self, model, dls):
                self.model = model
                self.dls = dls
                self.path = Path(MODEL_PATH)
                
            def predict(self, img):
                """Make a prediction for a single image"""
                self.model.eval()
                # Convert to tensor if it's not already
                if not isinstance(img, torch.Tensor):
                    # Apply transforms
                    transform = transforms.Compose([
                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img = transform(img).unsqueeze(0)
                
                img = img.to(DEVICE)
                with torch.no_grad():
                    output = self.model(img)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    _, predicted = torch.max(probs, 1)
                    pred_idx = predicted.item()
                    confidence = probs[0, pred_idx].item()
                    
                return self.dls.vocab[pred_idx], confidence, probs.cpu().numpy()[0]
        
        return PyTorchLearner(trained_model, dls)
    
    # Original FastAI implementation
    log_info("Using FastAI implementation for training")
    
    # Define a Top-K accuracy metric that works reliably
    def top_k_accuracy(inp, targ, k=3):
        "Computes the Top-k accuracy"
        inp = inp.argmax(dim=-1) if not isinstance(inp, torch.LongTensor) else inp
        preds = torch.topk(inp, k, dim=-1)[1]
        correct = preds == targ.unsqueeze(-1)
        return correct.any(dim=-1).float().mean()
    
    # Create callback for Top-3 accuracy
    class Top3Accuracy(Callback):
        def after_epoch(self):
            # Safer approach that works across FastAI versions
            if hasattr(self.learn, 'recorder') and hasattr(self.learn.recorder, 'metrics'):
                # Calculate top-3 accuracy
                dl = self.learn.dls.valid
                acc = 0
                total = 0
                self.learn.model.eval()
                
                with torch.no_grad():
                    for batch in dl:
                        # Safely handle batch data
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            x, y = batch[:2]  # Get input and target
                        else:
                            # Unexpected batch format
                            continue
                        
                        # This makes it more robust
                        if isinstance(x, tuple): 
                            x = x[0]
                            
                        # Get predictions
                        preds = self.learn.model(x)
                        
                        # Get top 3 indices
                        _, top3_idx = torch.topk(preds, 3, dim=1)
                        
                        # Compare with targets
                        targets = y
                        # Convert targets to match prediction format if needed
                        if len(targets.shape) > 1 and targets.shape[1] > 1:
                            targets = targets.argmax(dim=1)
                            
                        # Count correct predictions
                        for i, target in enumerate(targets):
                            if target.item() in top3_idx[i]:
                                acc += 1
                        total += len(targets)
                
                # Calculate and record metric
                if total > 0:
                    top3 = acc / total
                    # Add to recorder if available
                    if hasattr(self.learn.recorder, 'add_metrics'):
                        self.learn.recorder.add_metrics([top3])
                    elif hasattr(self.learn.recorder, 'metrics') and isinstance(self.learn.recorder.metrics, list):
                        self.learn.recorder.metrics.append(top3)
                        
                    # Log the result            
                    log_info(f"Top-3 Accuracy: {top3:.4f}")
    
    # Add class weights to reduce bias from imbalanced data
    if hasattr(dls, 'class_weights'):
        class_weights = dls.class_weights.to(DEVICE)
        # Create a weighted loss function 
        loss_func = CrossEntropyLossFlat(weight=class_weights)
        if LABEL_SMOOTHING > 0:
            loss_func = LabelSmoothingCrossEntropyFlat(
                eps=LABEL_SMOOTHING, weight=class_weights
            )
    else:
        loss_func = None  # Use default
    
    # Create first phase learner
    learn = create_vision_learner(
        dls, 
        metrics=[accuracy, top_k_accuracy], 
        loss_func=loss_func,
        cbs=[Top3Accuracy]
    )
    
    # Train the model with 1cycle policy for first phase
    log_info(f"Training first phase with {IMAGE_SIZE}px images for {NUM_EPOCHS} epochs")
    try:
        learn.fit_one_cycle(NUM_EPOCHS, LEARNING_RATE)
    except Exception as e:
        log_info(f"Error during training: {e}")
        # Try a simpler training approach
        log_info("Falling back to simpler training method")
        learn.fit(NUM_EPOCHS, LEARNING_RATE)
    
    # Save initial model
    learn.save('phase1')
    log_info("Saved phase 1 model")
    
    # Progressive resizing for better detail
    if USE_PROGRESSIVE_RESIZING and dls_final is not None:
        log_info(f"Fine-tuning with larger image size {FINAL_IMAGE_SIZE}")
        
        # Create a new learner with increased resolution
        learn_final = create_vision_learner(
            dls_final, 
            metrics=[accuracy, top_k_accuracy], 
            loss_func=loss_func,
            cbs=[Top3Accuracy]
        )
        
        # Load weights from first phase
        if not safe_load_model(learn_final, 'phase1'):
            log_info("Could not load phase 1 weights, starting from pretrained")
        
        # Train with discriminative learning rates
        try:
            learn_final.fit_one_cycle(
                FINE_TUNE_EPOCHS, 
                slice(LEARNING_RATE/10, LEARNING_RATE/3),
                cbs=MixUp(MIXUP_ALPHA) if USE_MIXUP else None
            )
        except Exception as e:
            log_info(f"Error during fine-tuning: {e}")
            # Try a simpler training approach
            log_info("Falling back to simpler training method for fine-tuning")
            learn_final.fit(FINE_TUNE_EPOCHS, LEARNING_RATE/5)
        
        # Save final model
        learn_final.save(MODEL_NAME.split('.')[0])
        log_info(f"Saved final model")
        
        return learn_final  # Return the fine-tuned learner
    else:
        # Save final model
        learn.save(MODEL_NAME.split('.')[0])
        log_info(f"Saved final model")
        
        return learn  # Return the initial learner

def predict_celebrity(model, image_path):
    """Make a prediction with better error handling and confidence calibration"""
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            return {'error': f"File not found: {image_path}"}
            
        # FastAI learner
        if hasattr(model, 'predict') and callable(model.predict):
            try:
                # Load the image for FastAI prediction
                img = PILImage.create(image_path)
                
                # Predict with the model
                pred_class, pred_idx, probs = model.predict(img)
                
                # Get confidence score
                confidence = probs[pred_idx].item()
                
                # Get top 3 predictions with confidence
                top3_indices = probs.topk(3)[1]
                top3_probs = probs[top3_indices]
                top3_predictions = []
                
                for i, idx in enumerate(top3_indices):
                    top3_predictions.append({
                        'class': model.dls.vocab[idx],
                        'confidence': top3_probs[i].item()
                    })
                
                return {
                    'prediction': pred_class,
                    'confidence': confidence,
                    'top3': top3_predictions
                }
            except Exception as e:
                return {'error': f"FastAI prediction error: {str(e)}"}
                
        # PyTorch model
        elif isinstance(model, torch.nn.Module):
            try:
                # Load and preprocess the image
                transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                img = Image.open(image_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                # Make prediction with test-time augmentation
                model.eval()
                with torch.no_grad():
                    if USE_TEST_TIME_AUGMENTATION:
                        # Use test-time augmentation for more robust predictions
                        probs = tta_inference(model, img_tensor)
                    else:
                        # Standard forward pass
                        output = model(img_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                    
                    # Get top 3 predictions
                    values, indices = torch.topk(probs, 3, dim=1)
                    
                    # Convert to numpy for easier handling
                    values = values.cpu().numpy()[0]
                    indices = indices.cpu().numpy()[0]
                    
                    # Top prediction
                    pred_idx = indices[0]
                    confidence = values[0]
                    
                    # Calculate calibrated confidence (optional)
                    # Temperature scaling would go here if implemented
                    
                    # Format results
                    top3_predictions = []
                    for i in range(3):
                        if i < len(indices):
                            top3_predictions.append({
                                'class': dls.vocab[indices[i]] if 'dls' in globals() else f"Class {indices[i]}",
                                'confidence': float(values[i])
                            })
                    
                    return {
                        'prediction': dls.vocab[pred_idx] if 'dls' in globals() else f"Class {pred_idx}",
                        'confidence': float(confidence),
                        'top3': top3_predictions
                    }
            except Exception as e:
                return {'error': f"PyTorch prediction error: {str(e)}"}
        else:
            return {'error': "Unsupported model type"}
    except Exception as e:
        return {'error': f"Prediction failed: {str(e)}"}

def predict_with_file(model_path, image_path, vocab=None):
    """Make a prediction using a saved model file"""
    try:
        # Determine if it's a FastAI model or PyTorch model
        if model_path.endswith('.pkl'):
            # Load FastAI model
            try:
                learn = load_learner(model_path)
                return predict_celebrity(learn, image_path)
            except Exception as e:
                log_info(f"Error loading FastAI model: {e}")
                # Fall back to PyTorch
                model_path = model_path.replace('.pkl', '.pth')
        
        # Try to load as PyTorch model
        if not os.path.exists(model_path):
            return {'error': f"Model not found: {model_path}"}
            
        # Get number of classes from vocab or default
        num_classes = len(vocab) if vocab else 30
        
        # Create model
        model = create_pytorch_model(ARCHITECTURE, num_classes=num_classes)
        
        # Load weights
        if not load_pytorch_model(model, model_path):
            return {'error': "Failed to load model weights"}
            
        # Load vocabulary if not provided
        if vocab is None:
            # Try to load from vocab file
            vocab_path = os.path.join(MODEL_PATH, 'vocab.pkl')
            if os.path.exists(vocab_path):
                try:
                    import pickle
                    with open(vocab_path, 'rb') as f:
                        vocab = pickle.load(f)
                except:
                    vocab = [f"Class {i}" for i in range(num_classes)]
            else:
                vocab = [f"Class {i}" for i in range(num_classes)]
                
        # Store vocab globally for prediction function
        global dls
        dls = type('', (), {})()  # Empty object
        dls.vocab = vocab
            
        # Make prediction
        return predict_celebrity(model, image_path)
    except Exception as e:
        return {'error': f"Prediction failed: {str(e)}"}

def main():
    """Execute the complete training and evaluation pipeline."""
    log_info("Starting celebrity recognition training pipeline")
    log_memory()
    
    # Make sure directories exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load and preprocess data
    df = load_identity_data()
    
    # Create data loaders with initial image size
    dls = setup_data_loaders(df, img_size=IMAGE_SIZE)
    log_info(f"Created data loaders with {len(dls.vocab)} classes")
    
    # Create a second set of data loaders for fine-tuning with larger images if needed
    dls_final = None
    if USE_PROGRESSIVE_RESIZING:
        dls_final = setup_data_loaders(df, img_size=FINAL_IMAGE_SIZE)
        log_info(f"Created secondary data loaders with image size {FINAL_IMAGE_SIZE}")
    
    # Train the model and get final learner
    learner = train_model(dls, dls_final)
    
    # Save the vocabulary for future use
    if USE_PURE_PYTORCH:
        # Save vocabulary for prediction
        import pickle
        with open(os.path.join(MODEL_PATH, 'vocab.pkl'), 'wb') as f:
            pickle.dump(dls.vocab, f)
        log_info("Saved vocabulary for future predictions")
    
    log_info("Training completed successfully")
    
    # Create a simple prediction script
    with open('predict.py', 'w') as f:
        f.write("""#!/usr/bin/env python
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
        print("\\nTop 3 predictions:")
        for i, pred in enumerate(result['top3']):
            print(f"{i+1}. {pred['class']}: {pred['confidence']:.4f}")

if __name__ == "__main__":
    main()
""")
    log_info("Created prediction script: predict.py")
    
    return learner

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_info(f"Error in main execution: {e}")
        log_info(traceback.format_exc()) 