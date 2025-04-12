#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastai.vision.all import *
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
import logging
import traceback
import sys
from types import SimpleNamespace

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
logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Reduce matplotlib logging noise

# Configuration
DATA_DIR = Path("img_align_celeba")
IDENTITY_FILE = "list_identity_celeba.txt"
MODEL_PATH = Path("models")
os.makedirs(MODEL_PATH, exist_ok=True)

# Simplified parameters
BATCH_SIZE = 4  # Even smaller batch size to reduce memory pressure
IMAGE_SIZE = 128  # Smaller size for faster processing and less memory
NUM_EPOCHS = 2  # Even fewer epochs for testing
MIN_IMAGES_PER_CELEB = 20
MAX_CELEBRITIES = 5  # Extremely limited number for diagnosis
ARCHITECTURE = 'resnet18'  # Smallest ResNet for compatibility and speed
USE_SIMPLE_TRAINING = False  # Skip FastAI training and use pure PyTorch implementation

# Diagnostic flags
VERBOSE = True
CHECK_DATALOADERS = True
SAVE_SAMPLE_IMAGES = True
LOG_MEMORY = True

# Set device globally to ensure consistency
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_info(msg):
    """Log information with timestamp"""
    logging.info(msg)
    if VERBOSE:
        print(msg)

def log_memory():
    """Log memory usage if GPU is available"""
    if not torch.cuda.is_available() or not LOG_MEMORY:
        return
    
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    cached = torch.cuda.memory_reserved() / (1024 * 1024)
    log_info(f"GPU Memory: {allocated:.2f}MB allocated, {cached:.2f}MB cached")

def load_identity_data():
    """Load a filtered subset of the identity data"""
    log_info("Loading identity data...")
    start_time = time.time()
    
    try:
        # Load the identity data
        df = pd.read_csv(IDENTITY_FILE, sep=r"\s+", skiprows=1, 
                        names=["image_id", "identity_name"])
        log_info(f"Loaded {len(df)} raw image identity records in {time.time() - start_time:.2f}s")
        
        # Add image paths for verification
        df['image_path'] = df['image_id'].apply(lambda x: DATA_DIR/x)
        
        # Only keep images that exist (in case some are missing)
        before_filter = len(df)
        df = df[df['image_path'].apply(lambda x: x.exists())]
        if before_filter > len(df):
            log_info(f"Filtered out {before_filter - len(df)} missing images")
        
        # Filter celebrities with too few images
        celebs_count = df['identity_name'].value_counts()
        valid_celebs = celebs_count[celebs_count >= MIN_IMAGES_PER_CELEB].index
        df = df[df['identity_name'].isin(valid_celebs)]
        log_info(f"After filtering for min {MIN_IMAGES_PER_CELEB} images: {len(df)} images of {len(valid_celebs)} celebrities")
        
        # Limit to top N celebrities
        if len(valid_celebs) > MAX_CELEBRITIES:
            top_celebs = df['identity_name'].value_counts().head(MAX_CELEBRITIES).index
            df = df[df['identity_name'].isin(top_celebs)]
            log_info(f"Limited to top {MAX_CELEBRITIES} celebrities: {len(df)} images")
        
        # Show class distribution
        class_counts = df['identity_name'].value_counts()
        log_info(f"Min images per class: {class_counts.min()}")
        log_info(f"Max images per class: {class_counts.max()}")
        log_info(f"Avg images per class: {class_counts.mean():.1f}")
        
        # Save sample images if enabled
        if SAVE_SAMPLE_IMAGES:
            os.makedirs("sample_images", exist_ok=True)
            # Sample a few images from each class for verification
            samples = []
            for celeb in df['identity_name'].unique():
                samples.extend(df[df['identity_name'] == celeb].sample(min(3, len(df[df['identity_name'] == celeb]))).index)
            
            sample_df = df.loc[samples]
            for i, row in sample_df.iterrows():
                import shutil
                dest = f"sample_images/{row['identity_name']}_{i}.jpg"
                shutil.copy(row['image_path'], dest)
            log_info(f"Saved {len(sample_df)} sample images to sample_images/")
        
        return df
    except Exception as e:
        logging.error(f"Error loading identity data: {e}")
        raise

def create_databunch(df):
    """Create and verify DataLoaders"""
    log_info("Creating databunch...")
    start_time = time.time()
    
    try:
        # Split data
        train_df, valid_df = train_test_split(
            df, test_size=0.2, stratify=df['identity_name'], random_state=42
        )
        
        log_info(f"Training set: {len(train_df)} images")
        log_info(f"Validation set: {len(valid_df)} images")
        
        # Use minimal transformations to avoid issues
        tfms = [Normalize.from_stats(*imagenet_stats)]
        
        # Define get_x and get_y functions at module level
        def get_x(r): return DATA_DIR/r['image_id']
        def get_y(r): return r['identity_name']
        
        # Create DataBlock with minimal augmentation
        celebrity_data = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=get_x,
            get_y=get_y,
            splitter=ColSplitter(),
            item_tfms=[Resize(IMAGE_SIZE)],  # Only resize, no other transforms
            batch_tfms=tfms
        )
        
        # Add is_valid column for splitting
        df['is_valid'] = np.where(df['image_id'].isin(valid_df['image_id']), True, False)
        
        # Create DataLoaders with conservative settings
        dls = celebrity_data.dataloaders(
            df, 
            bs=BATCH_SIZE,
            num_workers=0,  # Use 0 workers for simplicity and to avoid potential issues
            device=DEVICE  # Use the global device
        )
        
        log_info(f"Created databunch in {time.time() - start_time:.2f}s")
        
        # Verify DataLoaders
        if CHECK_DATALOADERS:
            log_info("Checking dataloaders...")
            # Try to get a batch
            try:
                xb, yb = dls.one_batch()
                log_info(f"Successfully retrieved a batch of shape {xb.shape}")
                log_info(f"Labels shape: {yb.shape}")
                log_info(f"Classes: {len(dls.vocab)} unique classes")
                log_info(f"Data device: {xb.device}")
                # Try to iterate through a few batches
                for i, (x, y) in enumerate(dls.train):
                    if i >= 2:  # Just check the first few batches
                        break
                    log_info(f"Batch {i}: {x.shape}, {y.shape}, device: {x.device}")
                log_info("Dataloader validation successful")
            except Exception as e:
                logging.error(f"Error checking dataloaders: {e}")
                raise
        
        return dls
    except Exception as e:
        logging.error(f"Error creating databunch: {e}")
        raise

# Helper function to convert FastAI tensor types to standard PyTorch tensors
def convert_fastai_tensor(t, keep_grad=True):
    """Convert a FastAI tensor to a standard PyTorch tensor while preserving gradient tracking if needed"""
    # Handle None values
    if t is None:
        return None
        
    # Special handling for TensorCategory which can be problematic
    if hasattr(t, '__class__') and str(t.__class__).find('TensorCategory') >= 0:
        # For TensorCategory, convert to standard long tensor (for class indices)
        # First get raw data
        if hasattr(t, 'data'):
            raw_data = t.data
        else:
            raw_data = t
            
        # Convert to standard PyTorch tensor of appropriate type
        if isinstance(raw_data, torch.Tensor):
            return raw_data.detach().clone().long()  # Force to long for class indices
        else:
            # Last resort - if we can get numpy data, convert through numpy
            try:
                if hasattr(t, 'numpy'):
                    numpy_data = t.numpy()
                elif hasattr(t, 'cpu'):
                    numpy_data = t.cpu().numpy()
                else:
                    numpy_data = np.array(t)
                return torch.tensor(numpy_data, dtype=torch.long)
            except:
                # If all else fails, just try direct conversion
                return torch.tensor(raw_data, dtype=torch.long)
        
    # Save original requires_grad state if possible
    requires_grad = False
    if keep_grad and hasattr(t, 'requires_grad'):
        requires_grad = t.requires_grad
        
    # For TensorImage, TensorCategory, or other fastai tensor types
    if hasattr(t, '__class__') and hasattr(t.__class__, '__name__') and (
        'Tensor' in t.__class__.__name__ or 
        hasattr(t, '__torch_function__') or
        hasattr(t, 'data')
    ):
        # Get raw tensor data if possible
        if hasattr(t, 'data'):
            t = t.data
        
        # Convert to regular tensor and preserve gradient properties
        if isinstance(t, torch.Tensor):
            result = t.detach().clone().to(dtype=t.dtype)
            if keep_grad and requires_grad:
                result.requires_grad_(True)
            return result
    
    # If it's already a standard tensor, just make sure it preserves gradient properties
    if isinstance(t, torch.Tensor):
        result = t.detach().clone()
        if keep_grad and requires_grad:
            result.requires_grad_(True)
        return result
        
    # Return as is if it's not a tensor
    return t

# Function to specifically handle validation tensors
def prepare_validation_batch(xb, yb, device):
    """Safely prepare validation tensors for loss calculation"""
    # Handle input tensor
    if not isinstance(xb, torch.Tensor) or hasattr(xb, '__torch_function__'):
        xb = convert_fastai_tensor(xb, keep_grad=False)
    
    # Handle target tensor - special case for TensorCategory
    if not isinstance(yb, torch.Tensor) or hasattr(yb, '__torch_function__'):
        # For TensorCategory, make sure we get class indices as long tensors
        if hasattr(yb, '__class__') and str(yb.__class__).find('TensorCategory') >= 0:
            # First try to get the raw tensor
            if hasattr(yb, 'data'):
                yb = yb.data
            
            # Make sure it's a long tensor for class indices
            if isinstance(yb, torch.Tensor):
                yb = yb.detach().clone().long()
            else:
                # Last resort conversion
                try:
                    yb = torch.tensor(yb, dtype=torch.long)
                except:
                    # Very last resort - through numpy
                    try:
                        if hasattr(yb, 'numpy'):
                            yb_np = yb.numpy() 
                        else:
                            yb_np = np.array(yb)
                        yb = torch.tensor(yb_np, dtype=torch.long)
                    except:
                        raise ValueError(f"Could not convert target tensor of type {type(yb)}")
        else:
            # Standard conversion for non-TensorCategory
            yb = convert_fastai_tensor(yb, keep_grad=False)
    
    # Move to device
    if hasattr(xb, 'device') and xb.device != device:
        xb = xb.to(device)
    if hasattr(yb, 'device') and yb.device != device:
        yb = yb.to(device)
    
    return xb, yb

# Modified training function to use raw tensors directly from custom dataloaders
def create_direct_dataloader(dls):
    """Create a standard PyTorch DataLoader that returns raw tensors instead of FastAI tensor types"""
    log_info("Creating direct dataloader that yields standard PyTorch tensors...")
    
    # Custom dataset that wraps FastAI DataLoaders but returns raw tensors
    class DirectTensorDataset:
        def __init__(self, fastai_dl):
            self.fastai_dl = fastai_dl
            self.length = len(fastai_dl)
            
        def __len__(self):
            return self.length
            
        def __iter__(self):
            # Create an iterator from the original dataloader
            fastai_iter = iter(self.fastai_dl)
            
            # Yield standard PyTorch tensors
            for batch in fastai_iter:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                    # Convert to standard tensors
                    x = convert_fastai_tensor(x, keep_grad=False)  # We'll set requires_grad later
                    y = convert_fastai_tensor(y, keep_grad=False)
                    # Force to device
                    if hasattr(x, 'device') and x.device != DEVICE:
                        x = x.to(DEVICE)
                    if hasattr(y, 'device') and y.device != DEVICE:
                        y = y.to(DEVICE)
                    yield x, y
                else:
                    # Handle unexpected batch format
                    log_info(f"Unexpected batch format: {type(batch)}")
                    yield batch
    
    # Create wrapper datasets
    train_ds = DirectTensorDataset(dls.train)
    valid_ds = DirectTensorDataset(dls.valid)
    
    # Create classes that hold these datasets with the same interface as fastai DataLoaders
    class DirectDataLoaders:
        def __init__(self, train_ds, valid_ds, vocab=None):
            self.train = train_ds
            self.valid = valid_ds
            self.vocab = vocab if vocab is not None else getattr(dls, 'vocab', None)
            
        def one_batch(self):
            """Get a single batch for testing"""
            return next(iter(self.train))
    
    return DirectDataLoaders(train_ds, valid_ds, dls.vocab)

# Function to create a pure PyTorch dataset from the original DataFrame
def create_pure_pytorch_datasets(dls):
    """Create pure PyTorch datasets without FastAI dependencies, using the original DataFrame"""
    log_info("Creating pure PyTorch datasets directly from data directory...")
    
    # Import required PyTorch components
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    
    # DirectImageDataset that loads images directly from disk
    class DirectImageDataset(Dataset):
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
    
    try:
        # Create a fresh copy of the original identity data
        log_info("Loading identity data for direct dataset creation...")
        df = pd.read_csv(IDENTITY_FILE, sep=r"\s+", skiprows=1, 
                        names=["image_id", "identity_name"])
        
        # Add image paths
        df['image_path'] = df['image_id'].apply(lambda x: str(DATA_DIR/x))
        
        # Filter to ensure files exist
        df = df[df['image_path'].apply(lambda x: os.path.exists(x))]
        log_info(f"Found {len(df)} images with existing paths")
        
        # Use the vocabulary from FastAI dataloaders
        vocab = dls.vocab
        log_info(f"Using vocabulary with {len(vocab)} classes")
        
        # Filter to only include classes in the vocabulary
        df = df[df['identity_name'].isin(vocab)]
        log_info(f"After filtering to classes in vocabulary: {len(df)} images")
        
        # Create label indices
        label_to_idx = {label: i for i, label in enumerate(vocab)}
        df['label_idx'] = df['identity_name'].apply(lambda x: label_to_idx[x])
        
        # Apply the same filters as in the original function
        # Filter celebrities with too few images
        celebs_count = df['identity_name'].value_counts()
        valid_celebs = celebs_count[celebs_count >= MIN_IMAGES_PER_CELEB].index
        df = df[df['identity_name'].isin(valid_celebs)]
        
        # Limit to top N celebrities
        if len(valid_celebs) > MAX_CELEBRITIES:
            top_celebs = df['identity_name'].value_counts().head(MAX_CELEBRITIES).index
            df = df[df['identity_name'].isin(top_celebs)]
            
        log_info(f"After applying all filters: {len(df)} images, {len(df['identity_name'].unique())} classes")
        
        # Split into train and validation
        from sklearn.model_selection import train_test_split
        train_df, valid_df = train_test_split(
            df, test_size=0.2, stratify=df['identity_name'], random_state=42
        )
        
        log_info(f"Training set: {len(train_df)} images")
        log_info(f"Validation set: {len(valid_df)} images")
        
        # Set up image transformations
        data_transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = DirectImageDataset(
            train_df['image_path'].tolist(),
            train_df['label_idx'].tolist(),
            transform=data_transforms
        )
        
        valid_dataset = DirectImageDataset(
            valid_df['image_path'].tolist(),
            valid_df['label_idx'].tolist(),
            transform=data_transforms
        )
        
        log_info(f"Created datasets with {len(train_dataset)} training and {len(valid_dataset)} validation samples")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=0  # Use 0 workers for simplicity
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=0  # Use 0 workers for simplicity
        )
        
        # Create a class similar to FastAI's DataLoaders for compatibility
        class PyTorchDataLoaders:
            def __init__(self, train_dl, valid_dl, vocab):
                self.train = train_dl
                self.valid = valid_dl
                self.vocab = vocab
                
            def one_batch(self):
                return next(iter(self.train))
                
        return PyTorchDataLoaders(train_loader, valid_loader, vocab)
        
    except Exception as e:
        log_info(f"Error creating pure PyTorch datasets: {e}")
        log_info(traceback.format_exc())
        return None

def train_model(dls):
    """Train a simple model and log progress"""
    log_info("Starting model training...")
    start_time = time.time()
    log_memory()
    
    try:
        # Skip the FastAI approach and go straight to pure PyTorch
        log_info("=" * 40)
        log_info("USING PURE PYTORCH TRAINING")
        log_info("=" * 40)
        
        # Create pure PyTorch dataloaders to avoid any FastAI tensor issues
        pytorch_dls = create_pure_pytorch_datasets(dls)
        
        if pytorch_dls is None:
            log_info("Falling back to direct dataloader conversion...")
            # Fall back to the direct dataloader if pure PyTorch dataset creation fails
            pytorch_dls = create_direct_dataloader(dls)
            
        # Create a pure PyTorch model
        try:
            # Get model architecture
            if ARCHITECTURE == 'resnet18':
                from torchvision.models import resnet18, ResNet18_Weights
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            elif ARCHITECTURE == 'resnet34':
                from torchvision.models import resnet34, ResNet34_Weights
                model = resnet34(weights=ResNet34_Weights.DEFAULT)
            else:
                # Default to ResNet18 if not recognized
                from torchvision.models import resnet18, ResNet18_Weights
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            
            # Get number of classes
            num_classes = len(dls.vocab)
            log_info(f"Model will classify {num_classes} classes")
            
            # Modify final layer for our classes
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            
            # Move to device
            model = model.to(DEVICE)
            
            # Set up optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # Create simple criterion
            criterion = nn.CrossEntropyLoss()
            
            # Train for a few epochs
            for epoch in range(NUM_EPOCHS):
                log_info(f"PyTorch Direct - Epoch {epoch+1}/{NUM_EPOCHS}")
                
                # Training phase
                model.train()
                train_losses = []
                
                for i, batch in enumerate(pytorch_dls.train):
                    if i % 5 == 0:
                        log_info(f"PyTorch batch {i}")
                    
                    try:
                        # Get and prepare data
                        inputs, targets = batch
                        
                        # Convert tensor types explicitly to ensure clean standard tensors
                        if not isinstance(inputs, torch.Tensor) or hasattr(inputs, '__torch_function__'):
                            # Get numpy and convert to clean tensor
                            inputs_np = inputs.cpu().numpy() if hasattr(inputs, 'cpu') else np.array(inputs)
                            inputs = torch.tensor(inputs_np, dtype=torch.float32)
                            
                        if not isinstance(targets, torch.Tensor) or hasattr(targets, '__torch_function__'):
                            # Get numpy and convert to clean tensor
                            targets_np = targets.cpu().numpy() if hasattr(targets, 'cpu') else np.array(targets)
                            targets = torch.tensor(targets_np, dtype=torch.long)
                        
                        # Check and fix tensor dimensions
                        if len(inputs.shape) != 4:
                            log_info(f"Fixing input shape: {inputs.shape} -> 4D tensor")
                            # Handle various dimension issues
                            if len(inputs.shape) == 5:  # Case: [1, 3, 3, 128, 128]
                                if inputs.shape[0] == 1 and inputs.shape[1] == 3 and inputs.shape[2] == 3:
                                    # This is likely [batch, channel, duplicate_channel, height, width]
                                    # Take just one set of channels
                                    inputs = inputs[:, :, 0, :, :]
                                else:
                                    # Generic reshape attempt - flatten all dimensions except last 2
                                    new_batch_size = np.prod(inputs.shape[:-3]).astype(int)
                                    new_channels = inputs.shape[-3]
                                    inputs = inputs.reshape(new_batch_size, new_channels, 
                                                          inputs.shape[-2], inputs.shape[-1])
                            elif len(inputs.shape) == 3:  # Case: [3, 128, 128]
                                # Missing batch dimension
                                inputs = inputs.unsqueeze(0)
                            else:
                                # Try to recover - check if this is a flattened batch
                                log_info(f"Unusual tensor shape: {inputs.shape}, attempting to reshape")
                                try:
                                    # Assume it might be a flattened tensor
                                    if inputs.shape[0] % 3 == 0:  # Multiple of channels
                                        batch_size = inputs.shape[0] // 3
                                        inputs = inputs.reshape(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
                                    else:
                                        # Last resort - treat as single example
                                        inputs = inputs.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)
                                except Exception as e:
                                    log_info(f"Failed to reshape tensor: {e}")
                                    raise ValueError(f"Cannot handle input tensor of shape {inputs.shape}")
                        
                        # Verify tensor shapes after fixing
                        if i % 5 == 0:
                            log_info(f"Input shape after fixing: {inputs.shape}")
                            log_info(f"Target shape: {targets.shape}")
                            
                        # Move to the right device
                        inputs = inputs.to(DEVICE)
                        targets = targets.to(DEVICE)
                        
                        # Forward pass - ensure inputs require gradients
                        inputs.requires_grad_(True)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        
                        # Verify tensor types if needed
                        if i % 20 == 0 and i < 3:
                            log_info(f"Training tensors - inputs: {inputs.shape}, dtype: {inputs.dtype}")
                            log_info(f"Training tensors - targets: {targets.shape}, dtype: {targets.dtype}")
                            log_info(f"Training tensors - outputs: {outputs.shape}, dtype: {outputs.dtype}")
                        
                        loss = criterion(outputs, targets)
                        # Backward and optimize
                        loss.backward()
                        optimizer.step()
                        
                        train_losses.append(loss.item())
                        if i % 10 == 0:
                            log_info(f"Batch {i} - Loss: {loss.item():.4f}")
                    except Exception as e:
                        log_info(f"Error in PyTorch training batch {i}: {e}")
                        log_info(traceback.format_exc())
                
                # Validation phase
                model.eval()
                val_losses = []
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for i, batch in enumerate(pytorch_dls.valid):
                        try:
                            # Get and prepare data
                            inputs, targets = batch
                            
                            # Convert tensor types explicitly to ensure clean standard tensors
                            if not isinstance(inputs, torch.Tensor) or hasattr(inputs, '__torch_function__'):
                                # Get numpy and convert to clean tensor
                                inputs_np = inputs.cpu().numpy() if hasattr(inputs, 'cpu') else np.array(inputs)
                                inputs = torch.tensor(inputs_np, dtype=torch.float32)
                                
                            if not isinstance(targets, torch.Tensor) or hasattr(targets, '__torch_function__'):
                                # Get numpy and convert to clean tensor
                                targets_np = targets.cpu().numpy() if hasattr(targets, 'cpu') else np.array(targets)
                                targets = torch.tensor(targets_np, dtype=torch.long)
                            
                            # Move to the right device
                            inputs = inputs.to(DEVICE)
                            targets = targets.to(DEVICE)
                            
                            # Verify tensor types if needed
                            if i % 20 == 0 and i < 3:
                                log_info(f"Validation tensors - inputs: {inputs.shape}, dtype: {inputs.dtype}")
                                log_info(f"Validation tensors - targets: {targets.shape}, dtype: {targets.dtype}")
                            
                            # Forward pass
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            
                            val_losses.append(loss.item())
                            
                            # Calculate accuracy
                            _, predicted = torch.max(outputs, 1)
                            total += targets.size(0)
                            correct += (predicted == targets).sum().item()
                        except Exception as e:
                            log_info(f"Error in PyTorch validation batch {i}: {e}")
                            if i < 5:  # Show details for first few errors
                                log_info(traceback.format_exc())
                
                # Log epoch stats
                if train_losses:
                    avg_train_loss = sum(train_losses) / len(train_losses)
                    log_info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
                
                if val_losses:
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    accuracy = correct / total if total > 0 else 0
                    log_info(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Save the model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/pure_pytorch_model.pth')
            log_info("Saved pure PyTorch model to models/pure_pytorch_model.pth")
            
            # Create a function to make predictions with the trained model
            def predict(img_path):
                from PIL import Image
                from torchvision import transforms
                
                # Define transformations (similar to what was used in training)
                transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # Load and preprocess the image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                # Make prediction
                model.eval()
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                
                # Get predicted class name
                idx_to_class = {i: c for i, c in enumerate(dls.vocab)}
                pred_class = idx_to_class[pred_idx.item()]
                confidence = conf.item()
                
                return pred_class, confidence
            
            # Write prediction function to a separate file
            with open('pytorch_predict.py', 'w') as f:
                f.write("""#!/usr/bin/env python
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
    print("\\nPredictions:")
    for i, (pred_class, confidence) in enumerate(results):
        print(f"{i+1}. {pred_class}: {confidence:.4f}")
""")
            log_info("Created standalone prediction script: pytorch_predict.py")
            
            # Save vocabulary for prediction
            import pickle
            with open('models/vocab.pkl', 'wb') as f:
                pickle.dump(dls.vocab, f)
            log_info("Saved vocabulary to models/vocab.pkl")
            
            # Return a SimpleNamespace object with the model and predict function
            return SimpleNamespace(model=model, predict=predict)
            
        except Exception as e:
            log_info(f"PyTorch training failed: {e}")
            log_info(traceback.format_exc())
            raise
            
    except Exception as e:
        logging.error(f"Error during training: {e}")
        logging.error(traceback.format_exc())
        raise

def main():
    """Main function to run training with detailed logging"""
    log_info("=" * 50)
    log_info("Starting simplified celebrity recognition training")
    log_info("=" * 50)
    
    # Log system info
    log_info(f"Python version: {sys.version}")
    log_info(f"PyTorch version: {torch.__version__}")
    try:
        import fastai
        log_info(f"FastAI version: {fastai.__version__}")
    except:
        log_info("Could not determine FastAI version")
    
    # Log hardware info
    if torch.cuda.is_available():
        log_info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        log_info(f"CUDA version: {torch.version.cuda}")
        log_info(f"Using device: {DEVICE}")
    else:
        log_info("CUDA not available, using CPU")
    
    try:
        # Log start time
        start_time = time.time()
        
        # Load data
        df = load_identity_data()
        log_info(f"Data loading completed in {time.time() - start_time:.2f}s")
        
        # Create dataloaders
        dls = create_databunch(df)
        log_info(f"DataLoaders creation completed in {time.time() - start_time:.2f}s")
        
        # Train model
        learn = train_model(dls)
        log_info(f"Total processing time: {time.time() - start_time:.2f}s")
        
        # Complete
        log_info("=" * 50)
        log_info("Training completed successfully")
        log_info("=" * 50)
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        log_info("=" * 50)
        log_info("Training failed - see log for details")
        log_info("=" * 50)
        
if __name__ == "__main__":
    main() 