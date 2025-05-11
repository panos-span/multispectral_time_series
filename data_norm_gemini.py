import json
import os
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import zarr # Make sure to install: pip install zarr
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split # pip install scikit-learn

# --- Configuration (Part 1 & 2) ---
BASE_DATA_PATH = "./timematch_data"
REGION = "denmark"
TILE = "32VNH"
YEAR = "2017"

NUM_PIXELS_TO_SAMPLE = 32
NORMALIZATION_DIVISOR = 10000.0
NUM_BANDS = 10
NUM_TIMESTEPS = 52
BATCH_SIZE = 32 # Example batch size
RANDOM_SEED = 42

DATA_ROOT = os.path.join(BASE_DATA_PATH, REGION, TILE, YEAR)
METADATA_PATH = os.path.join(DATA_ROOT, "meta")
ZARR_DATA_PATH = os.path.join(DATA_ROOT, "data")

LABELS_FILE = os.path.join(METADATA_PATH, "labels.json")
DATES_FILE = os.path.join(METADATA_PATH, "dates.json")
METADATA_PKL_FILE = os.path.join(METADATA_PATH, "metadata.pkl")

# --- Helper Functions (Part 1) ---
def load_json_file(file_path):
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

def load_pickle_file(file_path):
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

# --- Part 1: Data Exploration, Filtering, and Indexing ---
def preprocess_metadata(min_examples_threshold=200):
    print(f"Processing dataset for: {REGION} - {TILE} - {YEAR}")
    print("-" * 30)
    print("Running Part 1: Metadata Preprocessing...")

    labels_data = load_json_file(LABELS_FILE)
    if labels_data is None:
        raise FileNotFoundError(f"Could not load labels from {LABELS_FILE}")

    all_crop_labels = [labels_data[parcel_id] for parcel_id in labels_data.keys()]
    
    print("\n--- 1.1 Original Crop Categories and Counts ---")
    crop_counts = Counter(all_crop_labels)
    print(f"Found {len(crop_counts)} unique crop categories from {len(labels_data)} parcels.")

    print(f"\n--- 1.2 Filtering Categories (Min {min_examples_threshold} Examples) ---")
    crop_types_to_keep = [crop for crop, count in crop_counts.items() if count >= min_examples_threshold]
    
    filtered_labels_data = {}
    kept_parcel_ids = []
    for parcel_id, crop_name in labels_data.items():
        if crop_name in crop_types_to_keep:
            filtered_labels_data[parcel_id] = crop_name
            kept_parcel_ids.append(parcel_id)
            
    print(f"Kept {len(crop_types_to_keep)} categories.")
    print(f"Number of parcels after filtering: {len(filtered_labels_data)}")

    if not kept_parcel_ids:
        raise ValueError("No parcels left after filtering. Check threshold or data.")

    print("\n--- 1.3 New Indexing for Kept Categories ---")
    sorted_kept_crop_types = sorted(list(set(filtered_labels_data.values())))
    crop_to_idx = {crop_name: i for i, crop_name in enumerate(sorted_kept_crop_types)}
    idx_to_crop = {i: crop_name for crop_name, i in crop_to_idx.items()}
    num_classes = len(crop_to_idx)
    
    print("Crop to Index Mapping created.")
    print(f"Total number of classes after filtering and indexing: {num_classes}")
    print("-" * 30)
    return kept_parcel_ids, filtered_labels_data, crop_to_idx, idx_to_crop, num_classes

# --- Part 2: Dataset and DataLoader ---
def load_dates_as_day_of_year(dates_file_path, year_str):
    dates_data = load_json_file(dates_file_path)
    if dates_data is None:
        raise FileNotFoundError(f"Could not load dates from {dates_file_path}")
    
    day_of_year_list = []
    expected_year = int(year_str)
    for date_str in dates_data:
        dt_obj = datetime.strptime(str(date_str), "%Y%m%d")
        if dt_obj.year != expected_year:
            print(f"Warning: Date {date_str} (year {dt_obj.year}) is not in the expected year {expected_year}. Still calculating DOY for {expected_year}.")
        doy = dt_obj.timetuple().tm_yday
        day_of_year_list.append(doy)
    
    if len(day_of_year_list) != NUM_TIMESTEPS: # NUM_TIMESTEPS is 52
        print(f"Warning: Expected {NUM_TIMESTEPS} dates, but found {len(day_of_year_list)} in {dates_file_path}")

    print(f"Loaded {len(day_of_year_list)} dates and converted to Day-of-Year.")
    return torch.tensor(day_of_year_list, dtype=torch.long)


class SITSParcelDataset(Dataset):
    def __init__(self, parcel_ids, zarr_data_path, labels_dict, crop_to_idx,
                 dates_day_of_year, num_pixels_to_sample, normalization_divisor,
                 num_bands, num_timesteps, mode='train'):
        self.parcel_ids = parcel_ids
        self.zarr_data_path = zarr_data_path
        self.labels_dict = labels_dict
        self.crop_to_idx = crop_to_idx
        self.dates_day_of_year = dates_day_of_year
        self.num_pixels_to_sample = num_pixels_to_sample
        self.normalization_divisor = float(normalization_divisor)
        self.num_bands = num_bands
        self.num_timesteps = num_timesteps
        self.mode = mode

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError("Mode must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.parcel_ids)

    def __getitem__(self, idx):
        parcel_id_str = self.parcel_ids[idx]
        parcel_zarr_path = os.path.join(self.zarr_data_path, f"{parcel_id_str}.zarr")
        
        try:
            zarr_array_handle = zarr.open(parcel_zarr_path, mode='r')
            sits_data_permuted = np.array(zarr_array_handle) 
        except Exception as e:
            raise IOError(f"Could not load data for parcel {parcel_id_str} from {parcel_zarr_path}: {e}")

        # sits_data_permuted expected to be (T, C, S_actual) = (52, 10, num_pixels)
        # We need to permute it to (S_actual, T, C)
        if not (len(sits_data_permuted.shape) == 3 and \
                sits_data_permuted.shape[0] == self.num_timesteps and \
                sits_data_permuted.shape[1] == self.num_bands):
            raise ValueError(
                f"Data for parcel {parcel_id_str} (path: {parcel_zarr_path}) has unexpected "
                f"shape before permutation: {sits_data_permuted.shape}. "
                f"Expected 3D array with shape[0]={self.num_timesteps} (T) and shape[1]={self.num_bands} (C)."
            )
        
        # Permute from (T, C, S_actual) to (S_actual, T, C)
        sits_data = np.transpose(sits_data_permuted, (2, 0, 1)) 
        # Now sits_data should be (num_actual_pixels, num_timesteps, num_bands)
        
        num_actual_pixels = sits_data.shape[0]
        
        # This check is now for the permuted data
        if not (sits_data.shape[1] == self.num_timesteps and sits_data.shape[2] == self.num_bands):
             raise ValueError(
                f"Data for parcel {parcel_id_str} (path: {parcel_zarr_path}) has unexpected "
                f"shape AFTER permutation: {sits_data.shape}. "
                f"Expected shape[1]={self.num_timesteps} (T) and shape[2]={self.num_bands} (C)."
            )

        sampled_pixels_data = np.zeros((self.num_pixels_to_sample, self.num_timesteps, self.num_bands), dtype=np.float32)

        if self.mode == 'train':
            if num_actual_pixels == 0:
                 print(f"Warning: Parcel {parcel_id_str} has 0 actual pixels after permutation. Returning zero array.")
            elif num_actual_pixels >= self.num_pixels_to_sample:
                chosen_pixel_indices = np.random.choice(num_actual_pixels, self.num_pixels_to_sample, replace=False)
                sampled_pixels_data = sits_data[chosen_pixel_indices, :, :]
            else:
                chosen_pixel_indices = np.random.choice(num_actual_pixels, self.num_pixels_to_sample, replace=True)
                sampled_pixels_data = sits_data[chosen_pixel_indices, :, :]
        else: # 'val' or 'test' mode
            if num_actual_pixels == 0:
                 pass
            elif num_actual_pixels >= self.num_pixels_to_sample:
                sampled_pixels_data = sits_data[:self.num_pixels_to_sample, :, :]
            else:
                sampled_pixels_data[:num_actual_pixels, :, :] = sits_data
        
        normalized_pixels_data = sampled_pixels_data.astype(np.float32) / self.normalization_divisor
        
        crop_name = self.labels_dict[parcel_id_str]
        label = self.crop_to_idx[crop_name]
        
        pixels_tensor = torch.from_numpy(normalized_pixels_data).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return pixels_tensor, self.dates_day_of_year.clone(), label_tensor


if __name__ == "__main__":
    kept_parcel_ids_meta, filtered_labels_data_meta, crop_to_idx, \
    idx_to_crop, num_classes = preprocess_metadata()

    if not kept_parcel_ids_meta:
        print("No data to process after metadata preprocessing. Exiting.")
        exit()
        
    print("\nRunning Part 2: Dataset and DataLoader Creation...")
    
    print("\n--- Validating Zarr file shapes and accessibility (for permuted structure) ---")
    valid_zarr_parcel_ids = []
    invalid_zarr_details = []

    for p_id in kept_parcel_ids_meta: # Use IDs from metadata filtering
        parcel_zarr_path = os.path.join(ZARR_DATA_PATH, f"{p_id}.zarr")
        try:
            if not os.path.exists(parcel_zarr_path):
                invalid_zarr_details.append(f"Parcel {p_id}: Zarr file not found at {parcel_zarr_path}.")
                continue

            z_arr = zarr.open(parcel_zarr_path, mode='r')
            # Expected shape for z_arr.shape BEFORE permutation is (NUM_TIMESTEPS, NUM_BANDS, num_pixels)
            # So, z_arr.shape[0] should be NUM_TIMESTEPS (52)
            # z_arr.shape[1] should be NUM_BANDS (10)
            # z_arr.shape[2] should be > 0 (num_pixels)
            if len(z_arr.shape) == 3 and \
               z_arr.shape[0] == NUM_TIMESTEPS and \
               z_arr.shape[1] == NUM_BANDS and \
               z_arr.shape[2] > 0: # Number of pixels must be > 0
                valid_zarr_parcel_ids.append(p_id)
            elif z_arr.shape[2] == 0 : # Check if pixel dimension is zero
                 invalid_zarr_details.append(f"Parcel {p_id}: Zarr ({parcel_zarr_path}) has 0 pixels (shape[2]=0 in permuted view: {z_arr.shape}).")
            else:
                invalid_zarr_details.append(f"Parcel {p_id}: Zarr ({parcel_zarr_path}) has unexpected permuted shape: {z_arr.shape}. "
                                           f"Expected ({NUM_TIMESTEPS}, {NUM_BANDS}, >0).")
        except Exception as e:
            invalid_zarr_details.append(f"Parcel {p_id}: Could not open/read Zarr ({parcel_zarr_path}): {e}.")

    print(f"Found {len(valid_zarr_parcel_ids)} parcels with valid and accessible Zarr data (permuted structure check).")
    if invalid_zarr_details:
        print(f"Skipped {len(invalid_zarr_details)} parcels due to Zarr issues:")
        for detail in invalid_zarr_details[:5]: 
            print(f"  - {detail}")
        if len(invalid_zarr_details) > 5:
            print(f"  - ... and {len(invalid_zarr_details) - 5} more issues.")

    if not valid_zarr_parcel_ids:
        print("No valid Zarr files found after shape validation (permuted). Exiting.")
        exit()
    
    # Update kept_parcel_ids to only include valid ones for dataset creation
    kept_parcel_ids_for_dataset = valid_zarr_parcel_ids
    
    # Ensure filtered_labels_data is consistent
    current_filtered_labels_data = {pid: filtered_labels_data_meta[pid] for pid in kept_parcel_ids_for_dataset if pid in filtered_labels_data_meta}
    
    if len(kept_parcel_ids_for_dataset) != len(current_filtered_labels_data):
         print(f"Warning: Mismatch after Zarr validation. Parcel IDs: {len(kept_parcel_ids_for_dataset)}, Labels for them: {len(current_filtered_labels_data)}")
    
    if len(kept_parcel_ids_for_dataset) < len(kept_parcel_ids_meta):
         print(f"Effective number of parcels reduced from {len(kept_parcel_ids_meta)} to {len(kept_parcel_ids_for_dataset)} after Zarr validation.")
    
    if not kept_parcel_ids_for_dataset:
        print("No parcels remaining after Zarr validation. Exiting.")
        exit()
        
    dates_day_of_year = load_dates_as_day_of_year(DATES_FILE, YEAR)

    labels_for_stratification = [current_filtered_labels_data[pid] for pid in kept_parcel_ids_for_dataset]

    train_ids, val_ids = train_test_split(
        kept_parcel_ids_for_dataset,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels_for_stratification 
    )
    print(f"\nSplitting {len(kept_parcel_ids_for_dataset)} validated parcels into:")
    print(f"- Training set: {len(train_ids)} parcels")
    print(f"- Validation set: {len(val_ids)} parcels")

    print("\nCreating PyTorch Datasets...")
    train_dataset = SITSParcelDataset(
        parcel_ids=train_ids,
        zarr_data_path=ZARR_DATA_PATH,
        labels_dict=current_filtered_labels_data,
        crop_to_idx=crop_to_idx,
        dates_day_of_year=dates_day_of_year,
        num_pixels_to_sample=NUM_PIXELS_TO_SAMPLE,
        normalization_divisor=NORMALIZATION_DIVISOR,
        num_bands=NUM_BANDS,
        num_timesteps=NUM_TIMESTEPS,
        mode='train'
    )

    val_dataset = SITSParcelDataset(
        parcel_ids=val_ids,
        zarr_data_path=ZARR_DATA_PATH,
        labels_dict=current_filtered_labels_data,
        crop_to_idx=crop_to_idx,
        dates_day_of_year=dates_day_of_year,
        num_pixels_to_sample=NUM_PIXELS_TO_SAMPLE,
        normalization_divisor=NORMALIZATION_DIVISOR,
        num_bands=NUM_BANDS,
        num_timesteps=NUM_TIMESTEPS,
        mode='val' 
    )

    print("\nCreating PyTorch DataLoaders...")
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0 
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0 
    )

    print("\nDataLoaders created successfully.")
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")

    print("\nTesting DataLoaders by fetching a few batches...")
    print("\n--- Training DataLoader Example ---")
    try:
        for i, batch in enumerate(train_dataloader):
            pixels, dates, labels = batch
            print(f"Batch {i+1}:")
            print(f"  Pixels shape: {pixels.shape}") # Expected: (B, NUM_PIXELS_TO_SAMPLE, NUM_TIMESTEPS, NUM_BANDS)
            print(f"  Dates shape: {dates.shape}")   # Expected: (B, NUM_TIMESTEPS)
            print(f"  Labels shape: {labels.shape}") # Expected: (B,)
            if i == 1: break
    except Exception as e:
        print(f"Error during training DataLoader iteration: {e}")
        import traceback
        traceback.print_exc()
            
    print("\n--- Validation DataLoader Example ---")
    try:
        for i, batch in enumerate(val_dataloader):
            pixels, dates, labels = batch
            print(f"Batch {i+1}:")
            print(f"  Pixels shape: {pixels.shape}")
            print(f"  Dates shape: {dates.shape}")
            print(f"  Labels shape: {labels.shape}")
            if i == 1: break
    except Exception as e:
        print(f"Error during validation DataLoader iteration: {e}")
        import traceback
        traceback.print_exc()
            
    print("\nSetup complete. Ready for model training.")