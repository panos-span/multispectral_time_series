"""
TimeMatch Crop Classification - Data Preparation and Loading

This script prepares the TimeMatch dataset for crop classification:
1. Loads and filters crop categories with a minimum sample threshold
2. Creates a new indexing system for remaining categories
3. Implements a custom dataset with random pixel sampling for training
4. Sets up K-fold cross-validation data loaders

Dataset structure:
- Each parcel (agricultural plot) consists of a variable number of pixels
- For each pixel, we have a time series of multispectral data (10 channels)
- 52 acquisition dates per time series
- Each parcel is assigned a crop type label
"""

import json
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import zarr
from datetime import datetime
from collections import Counter
import random

# --- Configuration ---
BASE_DATA_PATH = "./timematch_data"
REGION = "denmark"
TILE = "32VNH"
YEAR = "2017"
NUM_PIXELS_SAMPLE = 32  # Fixed number of pixels to sample per parcel
NUM_FOLDS = 5  # For K-fold cross-validation
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42
MIN_EXAMPLES_THRESHOLD = 200  # Minimum examples per category to keep

# Construct paths
DATA_PATH = os.path.join(BASE_DATA_PATH, REGION, TILE, YEAR, "data")
METADATA_PATH = os.path.join(BASE_DATA_PATH, REGION, TILE, YEAR, "meta")
LABELS_FILE = os.path.join(METADATA_PATH, "labels.json")
METADATA_PKL_FILE = os.path.join(METADATA_PATH, "metadata.pkl")
DATES_FILE = os.path.join(METADATA_PATH, "dates.json")

# --- Helper Functions ---
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_json(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON from: {file_path}")
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {file_path}")
        return None

def load_pickle(file_path):
    """Load and parse a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded pickle from: {file_path}")
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return None
    except pickle.UnpicklingError:
        print(f"ERROR: Could not unpickle data from {file_path}")
        return None

def convert_dates_to_doy(dates):
    """Convert YYYYMMDD date strings to day-of-year (1-366)."""
    days_of_year = []
    for date_str in dates:
        date_str = str(date_str)  # Ensure it's a string
        try:
            date = datetime.strptime(date_str, '%Y%m%d')
            # Get day of year (1-366)
            days_of_year.append(date.timetuple().tm_yday)
        except ValueError:
            print(f"Warning: Could not parse date string {date_str}")
            days_of_year.append(0)  # Default value
    return np.array(days_of_year, dtype=np.int64)

# --- Data Preprocessing ---
def preprocess_dataset():
    """
    Preprocess the TimeMatch dataset:
    1. Load and analyze category distribution
    2. Filter categories with fewer than MIN_EXAMPLES_THRESHOLD examples
    3. Create new indexing for remaining categories
    4. Process acquisition dates
    
    Returns:
        kept_parcel_ids: List of parcel IDs to use
        filtered_labels: Dictionary mapping parcel_id -> crop_name
        crop_to_idx: Dictionary mapping crop_name -> integer label
        idx_to_crop: Dictionary mapping integer label -> crop_name
        num_classes: Number of unique classes
        dates_doy: Array of day-of-year values for each timestep
    """
    print(f"Processing dataset for: {REGION} - {TILE} - {YEAR}")
    print("-" * 30)

    # 1. Load labels data
    labels_data = load_json(LABELS_FILE)
    if labels_data is None:
        raise FileNotFoundError(f"Could not load labels from {LABELS_FILE}")

    # (Optional) Load parcel IDs from metadata.pkl
    metadata_content = load_pickle(METADATA_PKL_FILE)
    if metadata_content is None:
        print("Warning: Could not load metadata.pkl. Proceeding with labels.json keys only.")
        parcel_ids_from_meta = list(labels_data.keys())  # Fallback
    else:
        if 'parcels' in metadata_content:
            parcel_ids_from_meta = [str(p_id) for p_id in metadata_content['parcels']]
            print(f"Found {len(parcel_ids_from_meta)} parcel IDs in metadata.pkl.")
            missing_in_labels = [p_id for p_id in parcel_ids_from_meta if p_id not in labels_data]
            if missing_in_labels:
                print(f"Warning: {len(missing_in_labels)} parcel IDs from metadata.pkl are not in labels.json.")
        else:
            print("Warning: 'parcels' key not found in metadata.pkl. Using labels.json keys as parcel IDs.")
            parcel_ids_from_meta = list(labels_data.keys())  # Fallback

    # 2. Analyze crop categories
    all_crop_labels = [labels_data[parcel_id] for parcel_id in labels_data.keys()]
    
    print("\n--- 1. Original Crop Categories and Counts ---")
    crop_counts = Counter(all_crop_labels)
    print(f"Found {len(crop_counts)} unique crop categories.")
    for crop, count in crop_counts.most_common():
        print(f"- {crop}: {count} parcels")

    # 3. Filter categories with fewer than MIN_EXAMPLES_THRESHOLD examples
    print(f"\n--- 2. Filtering Categories (Min {MIN_EXAMPLES_THRESHOLD} Examples) ---")
    
    crop_types_to_keep = []
    crop_types_to_remove = []

    for crop, count in crop_counts.items():
        if count >= MIN_EXAMPLES_THRESHOLD:
            crop_types_to_keep.append(crop)
        else:
            crop_types_to_remove.append(crop)
            
    print(f"Keeping {len(crop_types_to_keep)} categories with >= {MIN_EXAMPLES_THRESHOLD} examples:")
    for crop in sorted(crop_types_to_keep):  # Sort for consistent output
        print(f"- {crop} (Count: {crop_counts[crop]})")
        
    if crop_types_to_remove:
        print(f"\nRemoving {len(crop_types_to_remove)} categories with < {MIN_EXAMPLES_THRESHOLD} examples:")
        for crop in sorted(crop_types_to_remove):
             print(f"- {crop} (Count: {crop_counts[crop]})")
    else:
        print("\nNo categories to remove based on the threshold.")

    # 4. Create filtered data
    filtered_labels_data = {}
    kept_parcel_ids = []

    for parcel_id, crop_name in labels_data.items():
        if crop_name in crop_types_to_keep:
            filtered_labels_data[parcel_id] = crop_name
            kept_parcel_ids.append(parcel_id)
            
    print(f"\nNumber of parcels after filtering: {len(filtered_labels_data)}")
    print(f"Number of unique crop types after filtering: {len(set(filtered_labels_data.values()))}")

    # 5. Define new indexing
    print("\n--- 3. New Indexing for Kept Categories ---")
    sorted_kept_crop_types = sorted(list(set(filtered_labels_data.values())))
    
    crop_to_idx = {crop_name: i for i, crop_name in enumerate(sorted_kept_crop_types)}
    idx_to_crop = {i: crop_name for crop_name, i in crop_to_idx.items()}
    
    print("Crop to Index Mapping:")
    for crop, idx in crop_to_idx.items():
        print(f"- '{crop}': {idx}")
        
    print("\nIndex to Crop Mapping:")
    for idx, crop in idx_to_crop.items():
        print(f"- {idx}: '{crop}'")
        
    num_classes = len(crop_to_idx)
    print(f"\nTotal number of classes after filtering and indexing: {num_classes}")

    # 6. Load date information
    dates_data = load_json(DATES_FILE)
    if dates_data is None:
        print("Warning: Could not load dates.json. Using dummy dates.")
        dates_doy = np.arange(52, dtype=np.int64)
    else:
        dates_doy = convert_dates_to_doy(dates_data)
        print(f"Loaded {len(dates_doy)} dates converted to day-of-year format.")
        print(f"First few dates (DoY): {dates_doy[:5]}")

    return kept_parcel_ids, filtered_labels_data, crop_to_idx, idx_to_crop, num_classes, dates_doy

# --- Dataset Class ---
class TimeMatchDataset(Dataset):
    """
    Dataset class for TimeMatch Satellite Image Time Series data.
    
    Features:
    - Loads data from zarr files for each parcel
    - Normalizes pixel values to [0, 1]
    - Random sampling of pixels during training
    - Deterministic sampling for validation/testing
    """
    
    def __init__(self, parcel_ids, labels_dict, crop_to_idx, data_path, dates_doy, 
                 is_training=True, num_pixels_sample=32):
        """
        Initialize the TimeMatch dataset.
        
        Args:
            parcel_ids: List of parcel IDs to include in the dataset
            labels_dict: Dictionary mapping parcel IDs to crop types
            crop_to_idx: Dictionary mapping crop types to integer indices
            data_path: Path to the data directory containing zarr files
            dates_doy: Array of day-of-year values for each timestep
            is_training: Whether this dataset is used for training (enables random sampling)
            num_pixels_sample: Number of pixels to sample per parcel
        """
        self.parcel_ids = parcel_ids
        self.labels_dict = labels_dict
        self.crop_to_idx = crop_to_idx
        self.data_path = data_path
        self.dates_doy = dates_doy
        self.is_training = is_training
        self.num_pixels_sample = num_pixels_sample
        
        # Max value for 16-bit integer, used for normalization
        self.max_pixel_value = 65535  # 2^16 - 1
        
        # Verify all parcels exist as zarr files
        self._verify_parcels()
        
    def _verify_parcels(self):
        """Verify that all parcel IDs exist as zarr files."""
        valid_parcel_ids = []
        for parcel_id in self.parcel_ids:
            zarr_path = os.path.join(self.data_path, f"{parcel_id}.zarr")
            if os.path.exists(zarr_path):
                valid_parcel_ids.append(parcel_id)
        
        print(f"Found {len(valid_parcel_ids)}/{len(self.parcel_ids)} valid parcel zarr files.")
        self.parcel_ids = valid_parcel_ids
    
    def __len__(self):
        return len(self.parcel_ids)
    
    def __getitem__(self, idx):
        """
        Get a parcel by index.
        
        Returns:
            A dictionary containing:
                - pixels: Tensor of shape (T, C, S) for timesteps, channels, sampled pixels
                - valid_pixels: Tensor of shape (T, S) indicating valid pixels
                - positions: Tensor of shape (T,) containing day-of-year values
                - label: Integer class label
                - parcel_id: Original parcel ID (for reference)
        """
        parcel_id = self.parcel_ids[idx]
        crop_type = self.labels_dict[parcel_id]
        label = self.crop_to_idx[crop_type]
        
        # Load parcel data from zarr file
        zarr_path = os.path.join(self.data_path, f"{parcel_id}.zarr")
        z = zarr.open(zarr_path, mode='r')
        
        # z.shape should be (T, C, S) for timesteps, channels, pixels
        T, C, S = z.shape
        
        # Sample pixels if training, or use all pixels if validation/testing
        if self.is_training:
            # Random sampling of pixels for training
            if S <= self.num_pixels_sample:
                # If we have fewer pixels than we want to sample, take all and repeat some
                indices = list(range(S))
                # Repeat first pixel if needed
                indices = indices + [0] * (self.num_pixels_sample - S)
                sampled_pixels = z[:, :, indices]
                valid_mask = np.ones((T, self.num_pixels_sample), dtype=np.float32)
                valid_mask[:, S:] = 0  # Mark repeated pixels as invalid
            else:
                # Randomly sample pixels
                indices = np.random.choice(S, self.num_pixels_sample, replace=False)
                sampled_pixels = z[:, :, indices]
                valid_mask = np.ones((T, self.num_pixels_sample), dtype=np.float32)
        else:
            # For validation/testing, use all pixels if possible (or a deterministic subset)
            if S <= self.num_pixels_sample:
                # If we have fewer pixels than expected, pad with zeros
                sampled_pixels = np.zeros((T, C, self.num_pixels_sample), dtype=np.float32)
                sampled_pixels[:, :, :S] = z[:]
                valid_mask = np.zeros((T, self.num_pixels_sample), dtype=np.float32)
                valid_mask[:, :S] = 1  # Mark original pixels as valid
            else:
                # If we have more pixels than expected, use a deterministic subset
                # For validation/testing, we want consistent results
                indices = np.linspace(0, S-1, self.num_pixels_sample, dtype=int)
                sampled_pixels = z[:, :, indices]
                valid_mask = np.ones((T, self.num_pixels_sample), dtype=np.float32)
        
        # Normalize data (scale to [0, 1])
        pixels = np.clip(sampled_pixels, 0, self.max_pixel_value).astype(np.float32) / self.max_pixel_value
        
        # Convert to PyTorch tensors
        pixels_tensor = torch.from_numpy(pixels)
        valid_mask_tensor = torch.from_numpy(valid_mask)
        dates_tensor = torch.from_numpy(self.dates_doy).long()
        
        return {
            'pixels': pixels_tensor,              # Shape: (T, C, S)
            'valid_pixels': valid_mask_tensor,    # Shape: (T, S)
            'positions': dates_tensor,            # Shape: (T,)
            'label': torch.tensor(label).long(),  # Scalar
            'parcel_id': parcel_id                # For reference
        }

# --- Data Loader Functions ---
def create_k_fold_loaders(parcel_ids, labels_dict, crop_to_idx, data_path, dates_doy, 
                         num_folds=5, batch_size=32, num_pixels_sample=32, num_workers=4, 
                         random_seed=42):
    """
    Create K DataLoaders for K-fold cross-validation.
    
    Args:
        parcel_ids: List of parcel IDs to include
        labels_dict: Dictionary mapping parcel IDs to crop types
        crop_to_idx: Dictionary mapping crop types to integer indices
        data_path: Path to the data directory containing zarr files
        dates_doy: Array of day-of-year values for each timestep
        num_folds: Number of folds for cross-validation
        batch_size: Batch size for DataLoader
        num_pixels_sample: Number of pixels to sample per parcel
        num_workers: Number of workers for DataLoader
        random_seed: Random seed for reproducibility
    
    Returns:
        A list of tuples (train_loader, val_loader) for each fold
    """
    # Set random seed for reproducibility
    set_random_seeds(random_seed)
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    
    # Convert parcel_ids to indices for KFold
    indices = np.arange(len(parcel_ids))
    
    fold_loaders = []
    
    # Create full dataset without sampling (we'll use subsets for train/val)
    full_dataset = TimeMatchDataset(
        parcel_ids=parcel_ids,
        labels_dict=labels_dict,
        crop_to_idx=crop_to_idx,
        data_path=data_path,
        dates_doy=dates_doy,
        is_training=False,  # Will be overridden for training subsets
        num_pixels_sample=num_pixels_sample
    )
    
    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        print(f"\nCreating loaders for fold {fold+1}/{num_folds}")
        
        # Create training subset with random sampling
        train_subset = Subset(full_dataset, train_indices)
        # We need to override is_training for the training subset
        train_subset.dataset.is_training = True
        
        # Create validation subset without random sampling
        val_subset = Subset(full_dataset, val_indices)
        # Ensure validation set has is_training=False
        val_subset.dataset.is_training = False
        
        print(f"  Training set: {len(train_subset)} parcels")
        print(f"  Validation set: {len(val_subset)} parcels")
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders

def test_data_loader(train_loader, val_loader, idx_to_crop):
    """Test the data loader by examining a batch."""
    print(f"Number of batches in training loader: {len(train_loader)}")
    print(f"Number of batches in validation loader: {len(val_loader)}")
    
    # Get one batch from the training loader to verify
    try:
        batch = next(iter(train_loader))
        pixels = batch['pixels']
        valid_pixels = batch['valid_pixels']
        positions = batch['positions']
        labels = batch['label']
        
        print(f"\nExample batch:")
        print(f"  Pixels shape: {pixels.shape}")  # Should be (batch_size, T, C, S)
        print(f"  Valid pixels shape: {valid_pixels.shape}")  # Should be (batch_size, T, S)
        print(f"  Positions shape: {positions.shape}")  # Should be (batch_size, T)
        print(f"  Labels shape: {labels.shape}")  # Should be (batch_size,)
        
        print(f"\nValue ranges:")
        print(f"  Pixels min/max: {pixels.min().item():.6f}/{pixels.max().item():.6f}")
        print(f"  Positions min/max: {positions.min().item()}/{positions.max().item()}")
        
        class_counts = Counter([idx_to_crop[label.item()] for label in labels])
        print(f"\nClasses in batch:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
            
        print("\nData loading setup completed successfully!")
        print("Ready to proceed with model implementation and training.")
        return True
        
    except Exception as e:
        print(f"Error testing data loader: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_random_seeds(RANDOM_SEED)
    
    # 1. Preprocess dataset
    kept_parcel_ids, filtered_labels, crop_to_idx, idx_to_crop, num_classes, dates_doy = preprocess_dataset()
    
    # 2. Create K-fold data loaders
    fold_loaders = create_k_fold_loaders(
        parcel_ids=kept_parcel_ids,
        labels_dict=filtered_labels,
        crop_to_idx=crop_to_idx,
        data_path=DATA_PATH,
        dates_doy=dates_doy,
        num_folds=NUM_FOLDS,
        batch_size=BATCH_SIZE,
        num_pixels_sample=NUM_PIXELS_SAMPLE,
        num_workers=NUM_WORKERS,
        random_seed=RANDOM_SEED
    )
    
    # 3. Test the data loaders
    print("\n--- Testing Data Loading ---")
    train_loader, val_loader = fold_loaders[0]  # First fold
    test_data_loader(train_loader, val_loader, idx_to_crop)