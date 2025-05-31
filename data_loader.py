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
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import functools

import numpy as np
import torch
import zarr
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

# --- Configuration ---
#BASE_DATA_PATH = "./timematch_data"
BASE_DATA_PATH = Path("timematch_data")
REGION = "denmark"
TILE = "32VNH"
YEAR = "2017"
NUM_PIXELS_SAMPLE = 32  # Fixed number of pixels to sample per parcel
NUM_FOLDS = 5  # For K-fold cross-validation
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42
MIN_EXAMPLES_THRESHOLD = 200  # Minimum examples per category to keep
INPUT_CHANNELS = 10 # <<< ENSURE THIS IS DEFINED HERE
NUM_TIMESTEPS = 52  # <<< ENSURE THIS IS DEFINED HERE
PIXEL_NORMALIZATION_MAX = 10000.0  # Max pixel value for normalization (65535.0 for 16-bit data)

# Construct paths
DATA_PATH = BASE_DATA_PATH / REGION / TILE / YEAR / "data"
METADATA_PATH = BASE_DATA_PATH / REGION / TILE / YEAR / "meta"
LABELS_FILE = METADATA_PATH / "labels.json"
METADATA_PKL_FILE = METADATA_PATH / "metadata.pkl"
DATES_FILE = METADATA_PATH / "dates.json"

# --- Helper Functions ---
def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_json(file_path: Path) -> Optional[Dict]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        print(f"Successfully loaded JSON from: {file_path}")
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {file_path}")
        return None


def load_pickle(file_path: Path) -> Optional[Dict]:
    """Load and parse a pickle file."""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"Successfully loaded pickle from: {file_path}")
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return None
    except pickle.UnpicklingError:
        print(f"ERROR: Could not unpickle data from {file_path}")
        return None


def convert_dates_to_doy(dates: List[str]) -> np.ndarray:
    """Convert YYYYMMDD date strings to day-of-year (1-366)."""
    days_of_year = []
    for date_str in dates:
        date_str = str(date_str)  # Ensure it's a string
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
            # Get day of year (1-366)
            days_of_year.append(date.timetuple().tm_yday)
        except ValueError:
            print(f"Warning: Could not parse date string {date_str}")
            days_of_year.append(0)  # Default value
    return np.array(days_of_year, dtype=np.int64)


# --- Data Preprocessing ---
def preprocess_dataset() -> Tuple[
    List[str],
    Dict[str, str],
    Dict[str, int],
    Dict[int, str],
    int,
    np.ndarray,
]:
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
        print(
            "Warning: Could not load metadata.pkl. Proceeding with labels.json keys only."
        )
        parcel_ids_from_meta = list(labels_data.keys())  # Fallback
    else:
        if "parcels" in metadata_content:
            parcel_ids_from_meta = [str(p_id) for p_id in metadata_content["parcels"]]
            print(f"Found {len(parcel_ids_from_meta)} parcel IDs in metadata.pkl.")
            missing_in_labels = [
                p_id for p_id in parcel_ids_from_meta if p_id not in labels_data
            ]
            if missing_in_labels:
                print(
                    f"Warning: {len(missing_in_labels)} parcel IDs from metadata.pkl are not in labels.json."
                )
        else:
            print(
                "Warning: 'parcels' key not found in metadata.pkl. Using labels.json keys as parcel IDs."
            )
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

    print(
        f"Keeping {len(crop_types_to_keep)} categories with >= {MIN_EXAMPLES_THRESHOLD} examples:"
    )
    for crop in sorted(crop_types_to_keep):  # Sort for consistent output
        print(f"- {crop} (Count: {crop_counts[crop]})")

    if crop_types_to_remove:
        print(
            f"\nRemoving {len(crop_types_to_remove)} categories with < {MIN_EXAMPLES_THRESHOLD} examples:"
        )
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
    print(
        f"Number of unique crop types after filtering: {len(set(filtered_labels_data.values()))}"
    )

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
        raise ValueError(
            "Dates data is required for processing. Please check the file."
        )
        # dates_doy = np.arange(52, dtype=np.int64)
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
                invalid_zarr_details.append(
                    f"Parcel {parcel_id}: Zarr file not found at {zarr_path}"
                )
                continue

            z_arr = zarr.open(zarr_path, mode="r")
            # Expected shape is (T, C, S) - time steps, channels, pixels
            if (
                len(z_arr.shape) == 3
                and z_arr.shape[0] == 52  # Number of time steps
                and z_arr.shape[1] == 10  # Number of channels
                and z_arr.shape[2] > 0
            ):  # Number of pixels > 0
                valid_parcel_ids.append(parcel_id)
            else:
                invalid_zarr_details.append(
                    f"Parcel {parcel_id}: Invalid shape {z_arr.shape}, expected (52, 10, >0)"
                )
        except Exception as e:
            invalid_zarr_details.append(
                f"Parcel {parcel_id}: Error loading zarr file: {e}"
            )

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

    return (
        kept_parcel_ids,
        filtered_labels_data,
        crop_to_idx,
        idx_to_crop,
        num_classes,
        dates_doy,
    )


# --- Dataset Class ---
class TimeMatchDataset(Dataset):
    """
    Memory-efficient dataset class for TimeMatch SITS data.

    Instead of preprocessing and storing sampled pixels,
    this dataset loads the full parcel data and defers sampling to batch creation time.
    """

    def __init__(
        self,
        parcel_ids: List[int],
        labels_dict: Dict[str, str],  # Mapping of parcel_id to crop type
        crop_to_idx: Dict[str, str],
        data_path: Path,
    ):
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

    def __len__(self) -> int:
        """Return the number of parcels in the dataset."""
        return len(self.parcel_ids)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
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
        # zarr_path = os.path.join(self.data_path, f"{parcel_id}.zarr")
        zarr_path = Path(self.data_path) / f"{parcel_id}.zarr"
        z = zarr.open(zarr_path, mode="r")

        # Return the raw zarr data, label and parcel_id
        # Sampling will be done in the collate function
        return {
            "zarr_data": np.array(z),  # Shape: (T, C, S) - time, channels, pixels
            "label": label,
            "parcel_id": parcel_id,
        }


def timematch_collate_fn(
    batch,
    dates_doy: np.ndarray,
    num_pixels_sample: int = NUM_PIXELS_SAMPLE,
    max_pixel_value: float = PIXEL_NORMALIZATION_MAX,
    is_training: bool = True,
) -> Dict[str, torch.Tensor]:
    """Custom collate function handling pixel sampling and padding. Constructs tensor directly."""
    # Filter out error samples first
    valid_batch = [s for s in batch if s['label'] != -1]
    actual_batch_size = len(valid_batch)

    if actual_batch_size == 0:
        # Return empty tensors if the whole batch was invalid
        return {
            "pixels": torch.empty((0, NUM_TIMESTEPS, INPUT_CHANNELS, num_pixels_sample), dtype=torch.float),
            "positions": torch.empty((0, NUM_TIMESTEPS), dtype=torch.long),
            "label": torch.empty((0,), dtype=torch.long),
            "parcel_id": [],
        }

    # Pre-allocate the final tensor
    pixels_tensor = torch.zeros(
        (actual_batch_size, NUM_TIMESTEPS, INPUT_CHANNELS, num_pixels_sample),
        dtype=torch.float
    )
    labels_list = []
    parcel_ids_list = []

    T = NUM_TIMESTEPS
    C = INPUT_CHANNELS

    for i, sample in enumerate(valid_batch):
        zarr_data = sample["zarr_data"]
        label = sample["label"]
        parcel_id = sample["parcel_id"]
        _T, _C, S = zarr_data.shape

        if _T != T or _C != C:
            print(f"Warning: Parcel {parcel_id} has unexpected shape ({_T}, {_C}, {S}). Expected ({T}, {C}, S). Skipping.")
            # Note: This sample was already filtered, but as a safeguard.
            # If it happens, the corresponding tensor slot will remain zero.
            labels_list.append(-1) # Or handle differently if needed
            parcel_ids_list.append(parcel_id)
            continue

        # Create a temporary numpy array for easier manipulation
        sampled_pixels_np = np.zeros((T, C, num_pixels_sample), dtype=np.float32)

        if S == 0:
            pass # sampled_pixels_np remains zeros
        elif S <= num_pixels_sample:
            sampled_pixels_np[:, :, :S] = zarr_data
            if S < num_pixels_sample:
                 sampled_pixels_np[:, :, S:] = np.broadcast_to(
                     zarr_data[:, :, 0:1], (T, C, num_pixels_sample - S)
                 ).copy() # Keep the copy for numpy safety
        else: # S > num_pixels_sample
            if is_training:
                indices = np.random.choice(S, num_pixels_sample, replace=False)
            else:
                indices = np.linspace(0, S - 1, num_pixels_sample, dtype=int)
            # Use .copy() when slicing numpy arrays if subsequent ops might modify views unexpectedly
            sampled_pixels_np = zarr_data[:, :, indices].copy()

        # Normalize
        normalized_pixels = sampled_pixels_np / max_pixel_value

        # Convert the processed numpy array for this sample to tensor and place it
        pixels_tensor[i] = torch.from_numpy(normalized_pixels).float() # <<< Place directly

        labels_list.append(label)
        parcel_ids_list.append(parcel_id)

    # Filter out any labels marked as -1 if skipping occurred
    valid_indices = [j for j, lbl in enumerate(labels_list) if lbl != -1]
    if len(valid_indices) < actual_batch_size:
         pixels_tensor = pixels_tensor[valid_indices]
         labels_tensor = torch.tensor([labels_list[j] for j in valid_indices], dtype=torch.long)
         parcel_ids_list = [parcel_ids_list[j] for j in valid_indices]
         actual_batch_size = len(valid_indices)
         if actual_batch_size == 0: # Check again if filtering removed everything
              # Return empty tensors
              return {
                    "pixels": torch.empty((0, T, C, num_pixels_sample), dtype=torch.float),
                    "positions": torch.empty((0, T), dtype=torch.long),
                    "label": torch.empty((0,), dtype=torch.long),
                    "parcel_id": [],
              }
    else:
         labels_tensor = torch.tensor(labels_list, dtype=torch.long)


    # We still clone here just to be absolutely sure memory is contiguous before pinning
    pixels_tensor = pixels_tensor.clone()

    positions_tensor = torch.from_numpy(dates_doy).long().unsqueeze(0).expand(actual_batch_size, -1)

    return {
        "pixels": pixels_tensor,
        "positions": positions_tensor,
        "label": labels_tensor,
        "parcel_id": parcel_ids_list,
    }


# --- K-fold Cross-validation ---
def create_k_fold_loaders(
    parcel_ids: List[str],
    labels_dict: Dict[str, str],
    crop_to_idx: Dict[str, str],
    data_path: Path,
    dates_doy: np.ndarray,
    num_folds: int = 5,
    batch_size: int = 32,
    num_pixels_sample: int = 32,
    num_workers: int = os.cpu_count(),
    random_seed: int = 42,
    max_pixel_value: int = 65535,  # 2^16 - 1 for 16-bit data,
) -> List[Tuple[DataLoader, DataLoader]]:
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
    set_random_seeds(random_seed)
    
    # We need an array of integer labels corresponding to parcel_ids for StratifiedKFold
    # The order of parcel_ids and y_labels must match.
    y_labels = np.array([crop_to_idx[labels_dict[pid]] for pid in parcel_ids])
    parcel_ids_array = np.array(parcel_ids) # Keep parcel_ids as an array for indexing
    
    # Create KFold object
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    # Create dataset
    full_dataset = TimeMatchDataset(
        parcel_ids=parcel_ids,
        labels_dict=labels_dict,
        crop_to_idx=crop_to_idx,
        data_path=data_path,
    )

    # Create fold loaders
    fold_loaders = []

    for fold, (train_indices, val_indices) in enumerate(skf.split(parcel_ids_array, y_labels)):
        print(f"\nCreating loaders for fold {fold+1}/{num_folds} (Stratified)")
        
        # Create subsets
        # These indices refer to the positions in parcel_ids_array (and thus full_dataset if parcel_ids was used to construct it)
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        # --- Print class distribution for verification (optional but good for debugging) ---
        train_subset_labels = [full_dataset[i]['label'] for i in train_indices]
        val_subset_labels = [full_dataset[i]['label'] for i in val_indices]
        print(f"  Fold {fold+1} Training Class Counts: {Counter(train_subset_labels)}")
        print(f"  Fold {fold+1} Validation Class Counts: {Counter(val_subset_labels)}")
        print("-"*50)

        print(f"  Training set: {len(train_subset)} parcels")
        print(f"  Validation set: {len(val_subset)} parcels")

        # Create collate functions with appropriate settings
        # --- Use functools.partial instead of lambda ---
        train_collate_fn = functools.partial(
            timematch_collate_fn, # The main collate function
            dates_doy=dates_doy,
            num_pixels_sample=num_pixels_sample,
            max_pixel_value=max_pixel_value,
            is_training=True
        )
        val_collate_fn = functools.partial(
            timematch_collate_fn, # The main collate function
            dates_doy=dates_doy,
            num_pixels_sample=num_pixels_sample,
            max_pixel_value=max_pixel_value,
            is_training=False
        )
        # --- End change ---


        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=train_collate_fn,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=val_collate_fn,
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
        pixels = batch["pixels"]
        valid_pixels = batch["valid_pixels"]
        positions = batch["positions"]
        labels = batch["label"]

        print("\nExample batch:")
        print(f"  Pixels shape: {pixels.shape}")  # Should be (B, T, C, S)
        print(f"  Valid pixels shape: {valid_pixels.shape}")  # Should be (B, T, S)
        print(f"  Positions shape: {positions.shape}")  # Should be (B, T)
        print(f"  Labels shape: {labels.shape}")  # Should be (B,)

        print("\nValue ranges:")
        print(f"  Pixels min/max: {pixels.min().item():.6f}/{pixels.max().item():.6f}")
        print(f"  Positions min/max: {positions.min().item()}/{positions.max().item()}")

        class_counts = Counter([idx_to_crop[label.item()] for label in labels])
        print("\nClasses in batch:")
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
    set_random_seeds(random_seed)

    # 1. Preprocess dataset
    (
        kept_parcel_ids,
        filtered_labels,
        crop_to_idx,
        idx_to_crop,
        num_classes,
        dates_doy,
    ) = preprocess_dataset()

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
        max_pixel_value=PIXEL_NORMALIZATION_MAX,  # Or 65535.0 if you prefer max value scaling : 65535  # 2^16 - 1 for 16-bit data
    )

    # 3. Test the data loaders
    print("\n--- Testing Data Loading ---")
    train_loader, val_loader = fold_loaders[0]  # First fold
    test_data_loader(train_loader, val_loader, idx_to_crop)
