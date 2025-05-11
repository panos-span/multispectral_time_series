"""
TimeMatch Crop Classification - Hybrid Data Loader

This script combines the best features of both implementations:
- Memory-efficient loading with custom collate function
- K-fold cross-validation setup
- Clean structure and organization
- Proper date handling

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
    random.seed(random_seed)
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
    5. Validate zarr files
    
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
        
    # 7. Validate zarr files
    print("\n--- 4. Validating Zarr Files ---")
    valid_parcel_ids = []
    invalid_zarr_details = []
    
    for parcel_id in kept_parcel_ids:
        zarr_path = os.path.join(DATA_PATH, f"{parcel_id}.zarr")
        try:
            if not os.path.exists(zarr_path):
                invalid_zarr_details.append(f"Parcel {parcel_id}: Zarr file not found at {zarr_path}")
                continue
                
            z_arr = zarr.open(zarr_path, mode='r')
            # Expected shape is (T, C, S) - time steps, channels, pixels
            if (len(z_arr.shape) == 3 and 
                z_arr.shape[0] == 52 and   # Number of time steps
                z_arr.shape[1] == 10 and   # Number of channels
                z_arr.shape[2] > 0):       # Number of pixels > 0
                valid_parcel_ids.append(parcel_id)
            else:
                invalid_zarr_details.append(f"Parcel {parcel_id}: Invalid shape {z_arr.shape}, expected (52, 10, >0)")
        except Exception as e:
            invalid_zarr_details.append(f"Parcel {parcel_id}: Error loading zarr file: {e}")
    
    print(f"Found {len(valid_parcel_ids)}/{len(kept_parcel_ids)} valid zarr files")
    if invalid_zarr_details:
        print(f"Skipped {len(invalid_zarr_details)} parcels due to zarr issues:")
        for detail in invalid_zarr_details[:5]:
            print(f"  - {detail}")
        if len(invalid_zarr_details) > 5:
            print(f"  - ... and {len(invalid_zarr_details) - 5} more issues")
            
    # Update kept_parcel_ids to only include valid zarr files
    kept_parcel_ids = valid_parcel_ids
    filtered_labels_data = {pid: filtered_labels_data[pid] for pid in kept_parcel_ids}
    
    print(f"\nFinal number of parcels with valid data: {len(kept_parcel_ids)}")

    return kept_parcel_ids, filtered_labels_data, crop_to_idx, idx_to_crop, num_classes, dates_doy

# --- Dataset Class ---
class TimeMatchDataset(Dataset):
    """
    Memory-efficient dataset class for TimeMatch SITS data.
    
    Instead of preprocessing and storing sampled pixels,
    this dataset loads the full parcel data and defers sampling to batch creation time.
    """
    
    def __init__(self, parcel_ids, labels_dict, crop_to_idx, data_path):
        """
        Initialize the TimeMatch dataset.
        
        Args:
            parcel_ids: List of parcel IDs to include in the dataset
            labels_dict: Dictionary mapping parcel IDs to crop types
            crop_to_idx: Dictionary mapping crop types to integer indices
            data_path: Path to the data directory containing zarr files
        """
        self.parcel_ids = parcel_ids
        self.labels_dict = labels_dict
        self.crop_to_idx = crop_to_idx
        self.data_path = data_path
    
    def __len__(self):
        return len(self.parcel_ids)
    
    def __getitem__(self, idx):
        """
        Get a parcel by index.
        
        Returns:
            zarr_data: The raw zarr array for the parcel
            label: Integer class label
            parcel_id: Original parcel ID (for reference)
        """
        parcel_id = self.parcel_ids[idx]
        crop_type = self.labels_dict[parcel_id]
        label = self.crop_to_idx[crop_type]
        
        # Load parcel data from zarr file
        zarr_path = os.path.join(self.data_path, f"{parcel_id}.zarr")
        z = zarr.open(zarr_path, mode='r')
        
        # Return the raw zarr data, label and parcel_id
        # Sampling will be done in the collate function
        return {
            'zarr_data': np.array(z),  # Shape: (T, C, S) - time, channels, pixels
            'label': label,
            'parcel_id': parcel_id
        }

def timematch_collate_fn(batch, dates_doy, num_pixels_sample=32, max_pixel_value=65535, is_training=True):
    """
    Custom collate function for TimeMatchDataset.
    
    Args:
        batch: List of samples returned by TimeMatchDataset.__getitem__
        dates_doy: Array of day-of-year values for each timestep
        num_pixels_sample: Number of pixels to sample per parcel
        max_pixel_value: Maximum pixel value for normalization
        is_training: Whether this batch is for training (uses random sampling)
        
    Returns:
        A dictionary containing:
            pixels: Tensor of shape (B, T, C, S) - batch, time, channels, sampled pixels
            valid_pixels: Tensor of shape (B, T, S) - batch, time, valid pixel mask
            positions: Tensor of shape (B, T) - batch, time steps as day-of-year
            labels: Tensor of shape (B) - batch labels
            parcel_ids: List of parcel IDs
    """
    pixels_list = []
    valid_pixels_list = []
    labels_list = []
    parcel_ids_list = []
    
    # Process each sample in the batch
    for sample in batch:
        zarr_data = sample['zarr_data']  # Shape: (T, C, S)
        label = sample['label']
        parcel_id = sample['parcel_id']
        
        T, C, S = zarr_data.shape
        
        # Sample or pad pixels
        sampled_pixels = np.zeros((T, C, num_pixels_sample), dtype=np.float32)
        valid_mask = np.zeros((T, num_pixels_sample), dtype=np.float32)
        
        if S == 0:
            # No pixels available, use zeros
            pass
        elif is_training:
            # Random sampling for training
            if S <= num_pixels_sample:
                # If not enough pixels, take all and repeat the first one
                sampled_pixels[:, :, :S] = zarr_data
                # Fill remaining slots by repeating the first pixel
                if S < num_pixels_sample:
                    sampled_pixels[:, :, S:] = np.broadcast_to(
                        zarr_data[:, :, 0:1], 
                        (T, C, num_pixels_sample - S)
                    )
                valid_mask[:, :S] = 1.0  # Mark original pixels as valid
            else:
                # Randomly sample pixels
                indices = np.random.choice(S, num_pixels_sample, replace=False)
                sampled_pixels = zarr_data[:, :, indices]
                valid_mask[:, :] = 1.0  # All pixels are valid
        else:
            # Deterministic sampling for validation/testing
            if S <= num_pixels_sample:
                # Take all available pixels
                sampled_pixels[:, :, :S] = zarr_data
                valid_mask[:, :S] = 1.0  # Mark original pixels as valid
            else:
                # Take evenly spaced pixels
                indices = np.linspace(0, S-1, num_pixels_sample, dtype=int)
                sampled_pixels = zarr_data[:, :, indices]
                valid_mask[:, :] = 1.0  # All pixels are valid
        
        # Normalize pixels
        normalized_pixels = sampled_pixels.astype(np.float32) / max_pixel_value
        
        # Add to lists
        pixels_list.append(normalized_pixels)
        valid_pixels_list.append(valid_mask)
        labels_list.append(label)
        parcel_ids_list.append(parcel_id)
    
    # Convert lists to tensors
    pixels_tensor = torch.from_numpy(np.stack(pixels_list, axis=0)).float()  # (B, T, C, S)
    valid_pixels_tensor = torch.from_numpy(np.stack(valid_pixels_list, axis=0)).float()  # (B, T, S)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)  # (B)
    
    # Create positions tensor (same for all samples in batch)
    positions_tensor = torch.from_numpy(dates_doy).long().unsqueeze(0).expand(len(batch), -1)  # (B, T)
    
    return {
        'pixels': pixels_tensor,
        'valid_pixels': valid_pixels_tensor,
        'positions': positions_tensor,
        'label': labels_tensor,
        'parcel_id': parcel_ids_list
    }

# --- K-fold Cross-validation ---
def create_k_fold_loaders(parcel_ids, labels_dict, crop_to_idx, data_path, dates_doy, 
                         num_folds=5, batch_size=32, num_pixels_sample=32, num_workers=4, 
                         random_seed=42, max_pixel_value=65535):
    """
    Create K DataLoaders for K-fold cross-validation with custom collate function.
    
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
        max_pixel_value: Maximum pixel value for normalization
    
    Returns:
        A list of tuples (train_loader, val_loader) for each fold
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    # Create KFold object
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    
    # Convert parcel_ids to indices for KFold
    indices = np.arange(len(parcel_ids))
    
    # Create dataset
    full_dataset = TimeMatchDataset(
        parcel_ids=parcel_ids,
        labels_dict=labels_dict,
        crop_to_idx=crop_to_idx,
        data_path=data_path
    )
    
    # Create fold loaders
    fold_loaders = []
    
    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        print(f"\nCreating loaders for fold {fold+1}/{num_folds}")
        
        # Get train and val parcel indices
        train_parcel_indices = indices[train_indices]
        val_parcel_indices = indices[val_indices]
        
        # Create subsets
        train_subset = Subset(full_dataset, train_parcel_indices)
        val_subset = Subset(full_dataset, val_parcel_indices)
        
        print(f"  Training set: {len(train_subset)} parcels")
        print(f"  Validation set: {len(val_subset)} parcels")
        
        # Create collate functions with appropriate settings
        train_collate_fn = lambda batch: timematch_collate_fn(
            batch, dates_doy, num_pixels_sample, max_pixel_value, is_training=True
        )
        
        val_collate_fn = lambda batch: timematch_collate_fn(
            batch, dates_doy, num_pixels_sample, max_pixel_value, is_training=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_collate_fn
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=val_collate_fn
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
        print(f"  Pixels shape: {pixels.shape}")           # Should be (B, T, C, S)
        print(f"  Valid pixels shape: {valid_pixels.shape}")  # Should be (B, T, S)
        print(f"  Positions shape: {positions.shape}")     # Should be (B, T)
        print(f"  Labels shape: {labels.shape}")           # Should be (B,)
        
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
        import traceback
        traceback.print_exc()
        return False

# --- Main Execution ---
if __name__ == "__main__":
    # Set random seeds for reproducibility
    random_seed = RANDOM_SEED
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
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
        random_seed=random_seed,
        max_pixel_value= 10000.0 # Or 65535.0 if you prefer max value scaling : 65535  # 2^16 - 1 for 16-bit data
    )
    
    # 3. Test the data loaders
    print("\n--- Testing Data Loading ---")
    train_loader, val_loader = fold_loaders[0]  # First fold
    test_data_loader(train_loader, val_loader, idx_to_crop)