"""
Enhanced TimeMatch Data Loader with Save/Load Functionality

This version allows you to:
1. Run data_loader.py once to preprocess and save everything
2. Load preprocessed data directly in timematch.py for faster iterations
"""

import json
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
from tqdm import tqdm

# --- Configuration ---
BASE_DATA_PATH = Path("timematch_data")
REGION = "denmark"
TILE = "32VNH"
YEAR = "2017"
NUM_PIXELS_SAMPLE = 32
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42
MIN_EXAMPLES_THRESHOLD = 200
INPUT_CHANNELS = 10
NUM_TIMESTEPS = 52

# --- New Normalization Configuration ---
NORMALIZATION_STRATEGY = "z_score"  # Options: "global", "z_score", "min_max", "robust"

# --- File paths for saving/loading preprocessed data ---
PREPROCESSED_DATA_DIR = BASE_DATA_PATH / REGION / TILE / YEAR / "preprocessed"
NORMALIZATION_STATS_FILE = PREPROCESSED_DATA_DIR / "normalization_stats.npz"
PREPROCESSED_METADATA_FILE = PREPROCESSED_DATA_DIR / "preprocessed_metadata.pkl"

# Construct paths
DATA_PATH = BASE_DATA_PATH / REGION / TILE / YEAR / "data"
METADATA_PATH = BASE_DATA_PATH / REGION / TILE / YEAR / "meta"
LABELS_FILE = METADATA_PATH / "labels.json"
METADATA_PKL_FILE = METADATA_PATH / "metadata.pkl"
DATES_FILE = METADATA_PATH / "dates.json"


class NormalizationComputer:
    """Computes and stores normalization statistics for multispectral data."""
    
    def __init__(self, strategy: str = "z_score"):
        """
        Initialize normalization computer.
        
        Args:
            strategy: Normalization strategy ("global", "z_score", "min_max", "robust")
        """
        self.strategy = strategy
        self.stats = None
    
    def compute_statistics(
        self, 
        parcel_ids: List[str], 
        data_path: Path,
        sample_size: Optional[int] = None,
        max_pixels_per_parcel: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute normalization statistics across the dataset.
        
        Args:
            parcel_ids: List of parcel IDs to process
            data_path: Path to zarr data files
            sample_size: Number of parcels to sample (None = all)
            max_pixels_per_parcel: Max pixels to sample per parcel for efficiency
            
        Returns:
            Dictionary containing normalization statistics
        """
        print(f"\nComputing {self.strategy} normalization statistics...")
        
        # Sample parcels if requested
        if sample_size and sample_size < len(parcel_ids):
            sampled_ids = random.sample(parcel_ids, sample_size)
            print(f"Sampling {sample_size} parcels out of {len(parcel_ids)}")
        else:
            sampled_ids = parcel_ids
            print(f"Processing all {len(parcel_ids)} parcels")
        
        # Collect data for statistics
        all_values = []
        
        for parcel_id in tqdm(sampled_ids, desc="Loading parcels for statistics"):
            zarr_path = data_path / f"{parcel_id}.zarr"
            
            try:
                z = zarr.open(zarr_path, mode="r")
                data = np.array(z)  # Shape: (T, C, S)
                
                # Sample pixels if too many
                T, C, S = data.shape
                if S > max_pixels_per_parcel:
                    pixel_indices = np.random.choice(S, max_pixels_per_parcel, replace=False)
                    data = data[:, :, pixel_indices]
                
                # Reshape to (N, C) where N = T * S
                data_reshaped = data.transpose(0, 2, 1).reshape(-1, C)  # (T*S, C)
                all_values.append(data_reshaped)
                
            except Exception as e:
                print(f"Warning: Could not load {parcel_id}: {e}")
                continue
        
        if not all_values:
            raise ValueError("No valid data loaded for computing statistics")
        
        # Concatenate all data
        all_data = np.concatenate(all_values, axis=0)  # (N_total, C)
        print(f"Loaded {all_data.shape[0]:,} samples across {all_data.shape[1]} bands")
        
        # Compute statistics based on strategy
        if self.strategy == "global":
            stats = {
                "max_value": np.array([10000.0])  # Original global normalization
            }
        
        elif self.strategy == "z_score":
            stats = {
                "mean": np.mean(all_data, axis=0),  # Shape: (C,)
                "std": np.std(all_data, axis=0),    # Shape: (C,)
            }
            # Prevent division by zero
            stats["std"] = np.maximum(stats["std"], 1e-8)
            
        elif self.strategy == "min_max":
            stats = {
                "min": np.min(all_data, axis=0),   # Shape: (C,)
                "max": np.max(all_data, axis=0),   # Shape: (C,)
            }
            # Prevent division by zero
            stats["range"] = np.maximum(stats["max"] - stats["min"], 1e-8)
            
        elif self.strategy == "robust":
            stats = {
                "median": np.median(all_data, axis=0),              # Shape: (C,)
                "q25": np.percentile(all_data, 25, axis=0),         # Shape: (C,)
                "q75": np.percentile(all_data, 75, axis=0),         # Shape: (C,)
            }
            stats["iqr"] = np.maximum(stats["q75"] - stats["q25"], 1e-8)
            
        else:
            raise ValueError(f"Unknown normalization strategy: {self.strategy}")
        
        self.stats = stats
        
        # Print statistics summary
        print(f"\nNormalization statistics ({self.strategy}):")
        for key, values in stats.items():
            if values.ndim == 1 and len(values) == INPUT_CHANNELS:
                print(f"  {key}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
            else:
                print(f"  {key}: {values}")
        
        return stats
    
    def save_statistics(self, filepath: Path):
        """Save computed statistics to file."""
        if self.stats is None:
            raise ValueError("No statistics computed. Call compute_statistics first.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(filepath, strategy=self.strategy, **self.stats)
        print(f"Saved normalization statistics to {filepath}")
    
    @classmethod
    def load_statistics(cls, filepath: Path) -> 'NormalizationComputer':
        """Load statistics from file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Statistics file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        strategy = str(data["strategy"])
        
        normalizer = cls(strategy=strategy)
        normalizer.stats = {key: data[key] for key in data.keys() if key != "strategy"}
        
        print(f"Loaded {strategy} normalization statistics from {filepath}")
        return normalizer
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply normalization to data.
        
        Args:
            data: Input data of shape (..., C) where C is number of channels
            
        Returns:
            Normalized data of same shape
        """
        if self.stats is None:
            raise ValueError("No statistics available. Compute or load statistics first.")
        
        if self.strategy == "global":
            return data / self.stats["max_value"]
        
        elif self.strategy == "z_score":
            return (data - self.stats["mean"]) / self.stats["std"]
        
        elif self.strategy == "min_max":
            return (data - self.stats["min"]) / self.stats["range"]
        
        elif self.strategy == "robust":
            return (data - self.stats["median"]) / self.stats["iqr"]
        
        else:
            raise ValueError(f"Unknown normalization strategy: {self.strategy}")


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
        date_str = str(date_str)
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
            days_of_year.append(date.timetuple().tm_yday)
        except ValueError:
            print(f"Warning: Could not parse date string {date_str}")
            days_of_year.append(0)
    return np.array(days_of_year, dtype=np.int64)


# --- Save/Load Functions ---
def save_preprocessed_data(
    kept_parcel_ids: List[str],
    filtered_labels: Dict[str, str],
    crop_to_idx: Dict[str, int],
    idx_to_crop: Dict[int, str],
    num_classes: int,
    dates_doy: np.ndarray,
    normalizer: NormalizationComputer,
) -> None:
    """Save all preprocessed data to files."""
    
    # Create directory
    PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save normalization statistics
    normalizer.save_statistics(NORMALIZATION_STATS_FILE)
    
    # Save metadata
    metadata = {
        "kept_parcel_ids": kept_parcel_ids,
        "filtered_labels": filtered_labels,
        "crop_to_idx": crop_to_idx,
        "idx_to_crop": idx_to_crop,
        "num_classes": num_classes,
        "dates_doy": dates_doy,
        "normalization_strategy": normalizer.strategy,
        "preprocessing_config": {
            "region": REGION,
            "tile": TILE,
            "year": YEAR,
            "min_examples_threshold": MIN_EXAMPLES_THRESHOLD,
            "input_channels": INPUT_CHANNELS,
            "num_timesteps": NUM_TIMESTEPS,
            "random_seed": RANDOM_SEED,
        }
    }
    
    with open(PREPROCESSED_METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\n✅ Saved all preprocessed data to {PREPROCESSED_DATA_DIR}")
    print(f"   - Normalization stats: {NORMALIZATION_STATS_FILE}")
    print(f"   - Metadata: {PREPROCESSED_METADATA_FILE}")
    print(f"   - Classes: {num_classes}")
    print(f"   - Parcels: {len(kept_parcel_ids)}")
    print(f"   - Normalization: {normalizer.strategy}")


def load_preprocessed_data() -> Tuple[
    List[str],
    Dict[str, str],
    Dict[str, int],
    Dict[int, str],
    int,
    np.ndarray,
    NormalizationComputer,
]:
    """Load all preprocessed data from files."""
    
    print(f"\n📁 Loading preprocessed data from {PREPROCESSED_DATA_DIR}")
    
    # Check if files exist
    if not PREPROCESSED_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessed metadata not found: {PREPROCESSED_METADATA_FILE}\n"
            f"Please run data_loader.py first to preprocess the data."
        )
    
    if not NORMALIZATION_STATS_FILE.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {NORMALIZATION_STATS_FILE}\n"
            f"Please run data_loader.py first to preprocess the data."
        )
    
    # Load metadata
    with open(PREPROCESSED_METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    
    # Load normalization statistics
    normalizer = NormalizationComputer.load_statistics(NORMALIZATION_STATS_FILE)
    
    # Verify strategy matches (optional warning)
    if normalizer.strategy != NORMALIZATION_STRATEGY:
        print(f"⚠️  Warning: Loaded normalization strategy '{normalizer.strategy}' "
              f"differs from config '{NORMALIZATION_STRATEGY}'")
        print(f"   Using loaded strategy: {normalizer.strategy}")
    
    # Extract data
    kept_parcel_ids = metadata["kept_parcel_ids"]
    filtered_labels = metadata["filtered_labels"]
    crop_to_idx = metadata["crop_to_idx"]
    idx_to_crop = metadata["idx_to_crop"]
    num_classes = metadata["num_classes"]
    dates_doy = metadata["dates_doy"]
    
    print(f"✅ Successfully loaded preprocessed data:")
    print(f"   - Classes: {num_classes}")
    print(f"   - Parcels: {len(kept_parcel_ids)}")
    print(f"   - Normalization: {normalizer.strategy}")
    print(f"   - Config: {metadata['preprocessing_config']}")
    
    return (
        kept_parcel_ids,
        filtered_labels,
        crop_to_idx,
        idx_to_crop,
        num_classes,
        dates_doy,
        normalizer,
    )


def check_preprocessed_data_exists() -> bool:
    """Check if preprocessed data exists."""
    return (PREPROCESSED_METADATA_FILE.exists() and 
            NORMALIZATION_STATS_FILE.exists())


# --- Enhanced Data Preprocessing ---
def preprocess_dataset_with_normalization() -> Tuple[
    List[str],
    Dict[str, str],
    Dict[str, int],
    Dict[int, str],
    int,
    np.ndarray,
    NormalizationComputer,
]:
    """
    Enhanced preprocessing that includes normalization statistics computation.
    
    Returns all the original outputs plus the normalization computer.
    """
    print(f"Processing dataset for: {REGION} - {TILE} - {YEAR}")
    print("-" * 50)

    # 1. Load labels data
    labels_data = load_json(LABELS_FILE)
    if labels_data is None:
        raise FileNotFoundError(f"Could not load labels from {LABELS_FILE}")

    # 2. Load metadata
    metadata_content = load_pickle(METADATA_PKL_FILE)
    if metadata_content is None:
        print("Warning: Could not load metadata.pkl. Using labels.json keys only.")
        parcel_ids_from_meta = list(labels_data.keys())
    else:
        if "parcels" in metadata_content:
            parcel_ids_from_meta = [str(p_id) for p_id in metadata_content["parcels"]]
            print(f"Found {len(parcel_ids_from_meta)} parcel IDs in metadata.pkl.")
        else:
            print("Warning: 'parcels' key not found in metadata.pkl.")
            parcel_ids_from_meta = list(labels_data.keys())

    # 3. Analyze and filter crop categories
    all_crop_labels = [labels_data[parcel_id] for parcel_id in labels_data.keys()]
    print("\n--- 1. Original Crop Categories and Counts ---")
    crop_counts = Counter(all_crop_labels)
    print(f"Found {len(crop_counts)} unique crop categories.")
    for crop, count in crop_counts.most_common():
        print(f"- {crop}: {count} parcels")

    print(f"\n--- 2. Filtering Categories (Min {MIN_EXAMPLES_THRESHOLD} Examples) ---")
    crop_types_to_keep = [crop for crop, count in crop_counts.items() 
                         if count >= MIN_EXAMPLES_THRESHOLD]
    
    print(f"Keeping {len(crop_types_to_keep)} categories:")
    for crop in sorted(crop_types_to_keep):
        print(f"- {crop} (Count: {crop_counts[crop]})")

    # 4. Create filtered data
    filtered_labels_data = {}
    kept_parcel_ids = []
    for parcel_id, crop_name in labels_data.items():
        if crop_name in crop_types_to_keep:
            filtered_labels_data[parcel_id] = crop_name
            kept_parcel_ids.append(parcel_id)

    print(f"\nNumber of parcels after filtering: {len(filtered_labels_data)}")

    # 5. Create indexing
    print("\n--- 3. New Indexing for Kept Categories ---")
    sorted_kept_crop_types = sorted(list(set(filtered_labels_data.values())))
    crop_to_idx = {crop_name: i for i, crop_name in enumerate(sorted_kept_crop_types)}
    idx_to_crop = {i: crop_name for crop_name, i in crop_to_idx.items()}
    num_classes = len(crop_to_idx)

    for crop, idx in crop_to_idx.items():
        print(f"- '{crop}': {idx}")

    # 6. Load dates
    dates_data = load_json(DATES_FILE)
    if dates_data is None:
        raise ValueError("Dates data is required for processing.")
    dates_doy = convert_dates_to_doy(dates_data)
    print(f"Loaded {len(dates_doy)} dates converted to day-of-year format.")

    # 7. Validate zarr files
    print("\n--- 4. Validating Zarr Files ---")
    valid_parcel_ids = []
    invalid_zarr_details = []

    for parcel_id in kept_parcel_ids:
        zarr_path = DATA_PATH / f"{parcel_id}.zarr"
        try:
            if not zarr_path.exists():
                invalid_zarr_details.append(f"Parcel {parcel_id}: Zarr file not found")
                continue

            z_arr = zarr.open(zarr_path, mode="r")
            if (len(z_arr.shape) == 3 and z_arr.shape[0] == 52 and 
                z_arr.shape[1] == 10 and z_arr.shape[2] > 0):
                valid_parcel_ids.append(parcel_id)
            else:
                invalid_zarr_details.append(
                    f"Parcel {parcel_id}: Invalid shape {z_arr.shape}"
                )
        except Exception as e:
            invalid_zarr_details.append(f"Parcel {parcel_id}: Error loading: {e}")

    print(f"Found {len(valid_parcel_ids)}/{len(kept_parcel_ids)} valid zarr files")
    if invalid_zarr_details:
        print(f"Skipped {len(invalid_zarr_details)} parcels due to zarr issues")

    kept_parcel_ids = valid_parcel_ids
    filtered_labels_data = {pid: filtered_labels_data[pid] for pid in kept_parcel_ids}

    # 8. Compute normalization statistics
    print("\n--- 5. Computing Normalization Statistics ---")
    
    normalizer = NormalizationComputer(strategy=NORMALIZATION_STRATEGY)
    
    # Use a subset for efficiency (you can adjust this)
    sample_size = min(len(kept_parcel_ids), 1000)  # Sample up to 1000 parcels
    
    normalizer.compute_statistics(
        parcel_ids=kept_parcel_ids,
        data_path=DATA_PATH,
        sample_size=sample_size,
        max_pixels_per_parcel=50  # Limit pixels per parcel for memory
    )

    print(f"\nFinal number of parcels with valid data: {len(kept_parcel_ids)}")

    return (
        kept_parcel_ids,
        filtered_labels_data,
        crop_to_idx,
        idx_to_crop,
        num_classes,
        dates_doy,
        normalizer,
    )


# --- Dataset Class (unchanged) ---
class TimeMatchDataset(Dataset):
    """Memory-efficient dataset class for TimeMatch SITS data."""

    def __init__(
        self,
        parcel_ids: List[int],
        labels_dict: Dict[str, str],
        crop_to_idx: Dict[str, str],
        data_path: Path,
    ):
        self.parcel_ids = parcel_ids
        self.labels_dict = labels_dict
        self.crop_to_idx = crop_to_idx
        self.data_path = data_path

    def __len__(self) -> int:
        return len(self.parcel_ids)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        parcel_id = self.parcel_ids[idx]
        crop_type = self.labels_dict[parcel_id]
        label = self.crop_to_idx[crop_type]

        zarr_path = Path(self.data_path) / f"{parcel_id}.zarr"
        z = zarr.open(zarr_path, mode="r")

        return {
            "zarr_data": np.array(z),
            "label": label,
            "parcel_id": parcel_id,
        }


def enhanced_timematch_collate_fn(
    batch,
    dates_doy: np.ndarray,
    normalizer: NormalizationComputer,
    num_pixels_sample: int = NUM_PIXELS_SAMPLE,
    is_training: bool = True,
) -> Dict[str, torch.Tensor]:
    """Enhanced collate function with sophisticated normalization."""
    # Filter out error samples
    valid_batch = [s for s in batch if s['label'] != -1]
    actual_batch_size = len(valid_batch)

    if actual_batch_size == 0:
        return {
            "pixels": torch.empty((0, NUM_TIMESTEPS, INPUT_CHANNELS, num_pixels_sample), dtype=torch.float),
            "positions": torch.empty((0, NUM_TIMESTEPS), dtype=torch.long),
            "label": torch.empty((0,), dtype=torch.long),
            "parcel_id": [],
        }

    # Pre-allocate tensor
    pixels_tensor = torch.zeros(
        (actual_batch_size, NUM_TIMESTEPS, INPUT_CHANNELS, num_pixels_sample),
        dtype=torch.float
    )
    labels_list = []
    parcel_ids_list = []

    T, C = NUM_TIMESTEPS, INPUT_CHANNELS

    for i, sample in enumerate(valid_batch):
        zarr_data = sample["zarr_data"]
        label = sample["label"]
        parcel_id = sample["parcel_id"]
        _T, _C, S = zarr_data.shape

        if _T != T or _C != C:
            print(f"Warning: Parcel {parcel_id} has unexpected shape ({_T}, {_C}, {S})")
            labels_list.append(-1)
            parcel_ids_list.append(parcel_id)
            continue

        # Pixel sampling (same logic as before)
        sampled_pixels_np = np.zeros((T, C, num_pixels_sample), dtype=np.float32)

        if S == 0:
            pass  # sampled_pixels_np remains zeros
        elif S <= num_pixels_sample:
            sampled_pixels_np[:, :, :S] = zarr_data
            if S < num_pixels_sample:
                sampled_pixels_np[:, :, S:] = np.broadcast_to(
                    zarr_data[:, :, 0:1], (T, C, num_pixels_sample - S)
                ).copy()
        else:  # S > num_pixels_sample
            if is_training:
                indices = np.random.choice(S, num_pixels_sample, replace=False)
            else:
                indices = np.linspace(0, S - 1, num_pixels_sample, dtype=int)
            sampled_pixels_np = zarr_data[:, :, indices].copy()

        # *** Enhanced Normalization ***
        # Reshape for per-band normalization: (T, C, S) -> (T*S, C)
        original_shape = sampled_pixels_np.shape
        reshaped_data = sampled_pixels_np.transpose(0, 2, 1).reshape(-1, C)  # (T*S, C)
        
        # Apply per-band normalization
        normalized_reshaped = normalizer.normalize(reshaped_data)  # (T*S, C)
        
        # Reshape back: (T*S, C) -> (T, C, S)
        normalized_pixels = normalized_reshaped.reshape(
            original_shape[0], original_shape[2], original_shape[1]
        ).transpose(0, 2, 1)  # (T, C, S)

        # Convert to tensor and store
        pixels_tensor[i] = torch.from_numpy(normalized_pixels).float()
        labels_list.append(label)
        parcel_ids_list.append(parcel_id)

    # Filter out any -1 labels
    valid_indices = [j for j, lbl in enumerate(labels_list) if lbl != -1]
    if len(valid_indices) < actual_batch_size:
        pixels_tensor = pixels_tensor[valid_indices]
        labels_tensor = torch.tensor([labels_list[j] for j in valid_indices], dtype=torch.long)
        parcel_ids_list = [parcel_ids_list[j] for j in valid_indices]
        actual_batch_size = len(valid_indices)
        
        if actual_batch_size == 0:
            return {
                "pixels": torch.empty((0, T, C, num_pixels_sample), dtype=torch.float),
                "positions": torch.empty((0, T), dtype=torch.long),
                "label": torch.empty((0,), dtype=torch.long),
                "parcel_id": [],
            }
    else:
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    pixels_tensor = pixels_tensor.clone()
    positions_tensor = torch.from_numpy(dates_doy).long().unsqueeze(0).expand(actual_batch_size, -1)

    return {
        "pixels": pixels_tensor,
        "positions": positions_tensor,
        "label": labels_tensor,
        "parcel_id": parcel_ids_list,
    }


# --- Enhanced K-fold Cross-validation ---
def create_enhanced_k_fold_loaders(
    parcel_ids: List[str],
    labels_dict: Dict[str, str],
    crop_to_idx: Dict[str, str],
    data_path: Path,
    dates_doy: np.ndarray,
    normalizer: NormalizationComputer,
    num_folds: int = 5,
    batch_size: int = 32,
    num_pixels_sample: int = 32,
    num_workers: int = 4,
    random_seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader]]:
    """Enhanced k-fold cross-validation with sophisticated normalization."""
    
    set_random_seeds(random_seed)
    
    y_labels = np.array([crop_to_idx[labels_dict[pid]] for pid in parcel_ids])
    parcel_ids_array = np.array(parcel_ids)
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    full_dataset = TimeMatchDataset(
        parcel_ids=parcel_ids,
        labels_dict=labels_dict,
        crop_to_idx=crop_to_idx,
        data_path=data_path,
    )

    fold_loaders = []

    for fold, (train_indices, val_indices) in enumerate(skf.split(parcel_ids_array, y_labels)):
        print(f"\nCreating enhanced loaders for fold {fold+1}/{num_folds}")
        
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        train_subset_labels = [full_dataset[i]['label'] for i in train_indices]
        val_subset_labels = [full_dataset[i]['label'] for i in val_indices]
        print(f"  Fold {fold+1} Training Class Counts: {Counter(train_subset_labels)}")
        print(f"  Fold {fold+1} Validation Class Counts: {Counter(val_subset_labels)}")

        # Enhanced collate functions with normalization
        train_collate_fn = functools.partial(
            enhanced_timematch_collate_fn,
            dates_doy=dates_doy,
            normalizer=normalizer,
            num_pixels_sample=num_pixels_sample,
            is_training=True
        )
        val_collate_fn = functools.partial(
            enhanced_timematch_collate_fn,
            dates_doy=dates_doy,
            normalizer=normalizer,
            num_pixels_sample=num_pixels_sample,
            is_training=False
        )

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


# --- Test function ---
def test_enhanced_data_loader(train_loader, val_loader, idx_to_crop, normalizer):
    """Test the enhanced data loader."""
    print(f"Number of batches in training loader: {len(train_loader)}")
    print(f"Number of batches in validation loader: {len(val_loader)}")
    print(f"Normalization strategy: {normalizer.strategy}")

    try:
        batch = next(iter(train_loader))
        pixels = batch["pixels"]
        positions = batch["positions"]
        labels = batch["label"]

        print("\nExample batch:")
        print(f"  Pixels shape: {pixels.shape}")
        print(f"  Positions shape: {positions.shape}")
        print(f"  Labels shape: {labels.shape}")

        print("\nValue ranges after enhanced normalization:")
        print(f"  Pixels min/max: {pixels.min().item():.6f}/{pixels.max().item():.6f}")
        print(f"  Pixels mean/std: {pixels.mean().item():.6f}/{pixels.std().item():.6f}")
        
        # Per-band statistics
        print("\nPer-band statistics (first 5 bands):")
        for i in range(min(5, pixels.shape[2])):
            band_data = pixels[:, :, i, :]  # (B, T, S)
            print(f"  Band {i}: mean={band_data.mean():.4f}, std={band_data.std():.4f}")

        class_counts = Counter([idx_to_crop[label.item()] for label in labels])
        print("\nClasses in batch:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")

        print("\nEnhanced data loading setup completed successfully!")
        return True

    except Exception as e:
        print(f"Error testing enhanced data loader: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- Main Execution ---
if __name__ == "__main__":
    set_random_seeds(RANDOM_SEED)
    print(f"Using normalization strategy: {NORMALIZATION_STRATEGY}")

    # 1. Enhanced preprocessing with normalization
    (
        kept_parcel_ids,
        filtered_labels,
        crop_to_idx,
        idx_to_crop,
        num_classes,
        dates_doy,
        normalizer,
    ) = preprocess_dataset_with_normalization()

    # 2. Save preprocessed data
    print("\n--- Saving Preprocessed Data ---")
    save_preprocessed_data(
        kept_parcel_ids=kept_parcel_ids,
        filtered_labels=filtered_labels,
        crop_to_idx=crop_to_idx,
        idx_to_crop=idx_to_crop,
        num_classes=num_classes,
        dates_doy=dates_doy,
        normalizer=normalizer,
    )

    print(f"\nDataset summary:")
    print(f"  Classes: {num_classes}")
    print(f"  Parcels: {len(kept_parcel_ids)}")
    print(f"  Normalization: {normalizer.strategy}")

    # 3. Create enhanced k-fold data loaders
    fold_loaders = create_enhanced_k_fold_loaders(
        parcel_ids=kept_parcel_ids,
        labels_dict=filtered_labels,
        crop_to_idx=crop_to_idx,
        data_path=DATA_PATH,
        dates_doy=dates_doy,
        normalizer=normalizer,
        num_folds=NUM_FOLDS,
        batch_size=BATCH_SIZE,
        num_pixels_sample=NUM_PIXELS_SAMPLE,
        num_workers=NUM_WORKERS,
        random_seed=RANDOM_SEED,
    )

    # 4. Test the enhanced data loaders
    print("\n--- Testing Enhanced Data Loading ---")
    train_loader, val_loader = fold_loaders[0]
    test_enhanced_data_loader(train_loader, val_loader, idx_to_crop, normalizer)
    
    print(f"\n🎉 Data preprocessing completed and saved!")
    print(f"   You can now run timematch.py with fast loading enabled.")