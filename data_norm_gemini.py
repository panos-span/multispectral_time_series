import json
import os
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Configuration ---
BASE_DATA_PATH = "./timematch_data"
REGION = "denmark"
TILE = "32VNH"
YEAR = "2017"

NUM_PIXELS_TO_SAMPLE_IN_BATCH = 32 # Renamed for clarity
NORMALIZATION_DIVISOR = 10000.0
NUM_BANDS = 10
NUM_TIMESTEPS = 52
BATCH_SIZE = 32
RANDOM_SEED = 42

DATA_ROOT = os.path.join(BASE_DATA_PATH, REGION, TILE, YEAR)
METADATA_PATH = os.path.join(DATA_ROOT, "meta")
ZARR_DATA_PATH = os.path.join(DATA_ROOT, "data")

LABELS_FILE = os.path.join(METADATA_PATH, "labels.json")
DATES_FILE = os.path.join(METADATA_PATH, "dates.json")

# --- Helper Functions ---
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

# --- Part 1: Metadata Preprocessing (Same as before) ---
def preprocess_metadata(min_examples_threshold=200):
    print(f"Processing dataset for: {REGION} - {TILE} - {YEAR}")
    print("-" * 30)
    print("Running Part 1: Metadata Preprocessing...")
    labels_data = load_json_file(LABELS_FILE)
    if labels_data is None:
        raise FileNotFoundError(f"Could not load labels from {LABELS_FILE}")
    all_crop_labels = [labels_data[parcel_id] for parcel_id in labels_data.keys()]
    crop_counts = Counter(all_crop_labels)
    print(f"\n--- 1.1 Original Crop Categories and Counts ---")
    print(f"Found {len(crop_counts)} unique crop categories from {len(labels_data)} parcels.")
    print(f"\n--- 1.2 Filtering Categories (Min {min_examples_threshold} Examples) ---")
    crop_types_to_keep = [crop for crop, count in crop_counts.items() if count >= min_examples_threshold]
    filtered_labels_data = {pid: name for pid, name in labels_data.items() if name in crop_types_to_keep}
    kept_parcel_ids = list(filtered_labels_data.keys())
    print(f"Kept {len(crop_types_to_keep)} categories.")
    print(f"Number of parcels after filtering: {len(filtered_labels_data)}")
    if not kept_parcel_ids:
        raise ValueError("No parcels left after filtering.")
    print("\n--- 1.3 New Indexing for Kept Categories ---")
    sorted_kept_crop_types = sorted(list(set(filtered_labels_data.values())))
    crop_to_idx = {name: i for i, name in enumerate(sorted_kept_crop_types)}
    idx_to_crop = {i: name for name, i in crop_to_idx.items()}
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
            print(f"Warning: Date {date_str} not in expected year {expected_year}.")
        doy = dt_obj.timetuple().tm_yday
        day_of_year_list.append(doy)
    if len(day_of_year_list) != NUM_TIMESTEPS:
        print(f"Warning: Expected {NUM_TIMESTEPS} dates, found {len(day_of_year_list)}.")
    print(f"Loaded {len(day_of_year_list)} dates as Day-of-Year.")
    return torch.tensor(day_of_year_list, dtype=torch.long)

class SITSFullParcelDataset(Dataset):
    def __init__(self, parcel_ids, zarr_data_path, labels_dict, crop_to_idx,
                 normalization_divisor, num_bands, num_timesteps):
        self.parcel_ids = parcel_ids
        self.zarr_data_path = zarr_data_path
        self.labels_dict = labels_dict
        self.crop_to_idx = crop_to_idx
        self.normalization_divisor = float(normalization_divisor)
        self.num_bands = num_bands
        self.num_timesteps = num_timesteps

    def __len__(self):
        return len(self.parcel_ids)

    def __getitem__(self, idx):
        parcel_id_str = self.parcel_ids[idx]
        parcel_zarr_path = os.path.join(self.zarr_data_path, f"{parcel_id_str}.zarr")
        
        try:
            zarr_array_handle = zarr.open(parcel_zarr_path, mode='r')
            # Expected loaded shape: (T, C, S_actual_pixels) = (52, 10, num_pixels)
            sits_data_tc_s = np.array(zarr_array_handle, dtype=np.float32) 
        except Exception as e:
            raise IOError(f"Could not load data for parcel {parcel_id_str} from {parcel_zarr_path}: {e}")

        if not (len(sits_data_tc_s.shape) == 3 and
                sits_data_tc_s.shape[0] == self.num_timesteps and
                sits_data_tc_s.shape[1] == self.num_bands):
            raise ValueError(
                f"Parcel {parcel_id_str} ({parcel_zarr_path}): Unexpected shape {sits_data_tc_s.shape}. "
                f"Expected ({self.num_timesteps}, {self.num_bands}, num_pixels)."
            )
        
        # Normalization (T, C, S_actual_pixels)
        normalized_sits_data = sits_data_tc_s / self.normalization_divisor
        
        crop_name = self.labels_dict[parcel_id_str]
        label = self.crop_to_idx[crop_name]
        
        # Return: (Time, Channels, All_Pixels_for_Parcel), label
        return torch.from_numpy(normalized_sits_data).float(), torch.tensor(label, dtype=torch.long)


def sitscustom_collate_fn(batch, num_pixels_to_sample, mode='train'):
    """
    Custom collate_fn for SITS data.
    Input 'batch': A list of tuples, where each tuple is (sits_data_tensor, label_tensor).
                   sits_data_tensor has shape (T, C, S_actual_pixels_for_this_parcel).
    Output:
        - A tensor of batched SITS data: (Batch_Size, T, C, num_pixels_to_sample)
        - A tensor of batched dates: (Batch_Size, T) - assuming dates are common per batch item
        - A tensor of batched labels: (Batch_Size,)
    """
    sits_data_list = []
    labels_list = []

    for item_sits_data, item_label in batch:
        # item_sits_data shape: (T, C, S_actual)
        num_actual_pixels = item_sits_data.shape[2]
        
        sampled_pixels_for_item = torch.zeros((NUM_TIMESTEPS, NUM_BANDS, num_pixels_to_sample), dtype=item_sits_data.dtype)

        if num_actual_pixels == 0:
            # print(f"Warning: Collate found an item with 0 actual pixels. Using zeros.")
            pass # sampled_pixels_for_item remains zeros
        elif mode == 'train':
            if num_actual_pixels >= num_pixels_to_sample:
                chosen_indices = torch.randperm(num_actual_pixels)[:num_pixels_to_sample]
            else: # Sample with replacement
                chosen_indices = torch.randint(0, num_actual_pixels, (num_pixels_to_sample,))
            sampled_pixels_for_item = item_sits_data[:, :, chosen_indices]
        else: # 'val' or 'test' - deterministic sampling
            if num_actual_pixels >= num_pixels_to_sample:
                sampled_pixels_for_item = item_sits_data[:, :, :num_pixels_to_sample]
            else: # Pad with zeros if not enough pixels
                sampled_pixels_for_item[:, :, :num_actual_pixels] = item_sits_data
        
        sits_data_list.append(sampled_pixels_for_item)
        labels_list.append(item_label)

    # Stack into batches
    # sits_data_list contains tensors of shape (T, C, num_pixels_to_sample)
    batched_sits_data = torch.stack(sits_data_list, dim=0) # (B, T, C, num_pixels_to_sample)
    batched_labels = torch.stack(labels_list, dim=0)       # (B,)
    
    return batched_sits_data, batched_labels


if __name__ == "__main__":
    kept_parcel_ids_meta, filtered_labels_data_meta, crop_to_idx, \
    idx_to_crop, num_classes = preprocess_metadata()

    if not kept_parcel_ids_meta:
        print("Exiting.")
        exit()
        
    print("\nRunning Part 2: Dataset and DataLoader Creation...")
    
    print("\n--- Validating Zarr file shapes (T, C, S_pixels) ---")
    valid_zarr_parcel_ids = []
    invalid_zarr_details = []

    for p_id in kept_parcel_ids_meta:
        parcel_zarr_path = os.path.join(ZARR_DATA_PATH, f"{p_id}.zarr")
        try:
            if not os.path.exists(parcel_zarr_path):
                invalid_zarr_details.append(f"Parcel {p_id}: Zarr not found {parcel_zarr_path}.")
                continue
            z_arr = zarr.open(parcel_zarr_path, mode='r')
            if len(z_arr.shape) == 3 and \
               z_arr.shape[0] == NUM_TIMESTEPS and \
               z_arr.shape[1] == NUM_BANDS and \
               z_arr.shape[2] > 0: # Num_pixels > 0
                valid_zarr_parcel_ids.append(p_id)
            elif z_arr.shape[2] == 0:
                 invalid_zarr_details.append(f"Parcel {p_id}: Zarr ({parcel_zarr_path}) has 0 pixels (shape[2]=0 in T,C,S view: {z_arr.shape}).")
            else:
                invalid_zarr_details.append(f"Parcel {p_id}: Zarr ({parcel_zarr_path}) unexpected shape {z_arr.shape}. Expected ({NUM_TIMESTEPS}, {NUM_BANDS}, >0).")
        except Exception as e:
            invalid_zarr_details.append(f"Parcel {p_id}: Error with Zarr ({parcel_zarr_path}): {e}.")

    print(f"Found {len(valid_zarr_parcel_ids)} parcels with valid Zarr data (T,C,S_pixels structure).")
    if invalid_zarr_details:
        print(f"Skipped {len(invalid_zarr_details)} parcels:")
        for detail in invalid_zarr_details[:5]: print(f"  - {detail}")
        if len(invalid_zarr_details) > 5: print(f"  - ... and {len(invalid_zarr_details) - 5} more.")

    if not valid_zarr_parcel_ids:
        print("No valid Zarr files found. Exiting.")
        exit()
    
    kept_parcel_ids_for_dataset = valid_zarr_parcel_ids
    current_filtered_labels_data = {pid: filtered_labels_data_meta[pid] for pid in kept_parcel_ids_for_dataset}
    
    if len(kept_parcel_ids_for_dataset) < len(kept_parcel_ids_meta):
         print(f"Effective parcels reduced from {len(kept_parcel_ids_meta)} to {len(kept_parcel_ids_for_dataset)}.")
    
    # --- Dates are loaded once, they will be added by the main training loop if needed ---
    # --- Or if dates need to be part of the batch, the collate_fn can be modified to accept and return them ---
    dates_day_of_year = load_dates_as_day_of_year(DATES_FILE, YEAR) # (T,)

    labels_for_stratification = [current_filtered_labels_data[pid] for pid in kept_parcel_ids_for_dataset]
    train_ids, val_ids = train_test_split(
        kept_parcel_ids_for_dataset, test_size=0.2, random_state=RANDOM_SEED, stratify=labels_for_stratification
    )
    print(f"\nSplitting {len(kept_parcel_ids_for_dataset)} parcels: Train: {len(train_ids)}, Val: {len(val_ids)}")

    print("\nCreating PyTorch Datasets...")
    train_dataset = SITSFullParcelDataset(
        parcel_ids=train_ids, zarr_data_path=ZARR_DATA_PATH, labels_dict=current_filtered_labels_data,
        crop_to_idx=crop_to_idx, normalization_divisor=NORMALIZATION_DIVISOR,
        num_bands=NUM_BANDS, num_timesteps=NUM_TIMESTEPS
    )
    val_dataset = SITSFullParcelDataset(
        parcel_ids=val_ids, zarr_data_path=ZARR_DATA_PATH, labels_dict=current_filtered_labels_data,
        crop_to_idx=crop_to_idx, normalization_divisor=NORMALIZATION_DIVISOR,
        num_bands=NUM_BANDS, num_timesteps=NUM_TIMESTEPS
    )

    # --- Create DataLoaders with the custom collate function ---
    print("\nCreating PyTorch DataLoaders with custom collate_fn...")
    train_collate_fn = lambda batch: sitscustom_collate_fn(batch, NUM_PIXELS_TO_SAMPLE_IN_BATCH, mode='train')
    val_collate_fn = lambda batch: sitscustom_collate_fn(batch, NUM_PIXELS_TO_SAMPLE_IN_BATCH, mode='val')

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=train_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=val_collate_fn
    )

    print("\nDataLoaders created.")
    print(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    print("\nTesting DataLoaders...")
    print("\n--- Training DataLoader Example ---")
    try:
        # The collate_fn now returns (batched_sits_data, batched_labels)
        # Dates are handled separately if needed by the model.
        for i, (batch_pixels, batch_labels) in enumerate(train_dataloader):
            print(f"Batch {i+1}:")
            print(f"  Pixels shape: {batch_pixels.shape}") # Expected: (B, T, C, NUM_PIXELS_TO_SAMPLE_IN_BATCH)
            print(f"  Labels shape: {batch_labels.shape}") # Expected: (B,)
            if i == 1: break
    except Exception as e:
        print(f"Error during training DataLoader iteration: {e}")
        import traceback
        traceback.print_exc()
            
    print("\n--- Validation DataLoader Example ---")
    try:
        for i, (batch_pixels, batch_labels) in enumerate(val_dataloader):
            print(f"Batch {i+1}:")
            print(f"  Pixels shape: {batch_pixels.shape}")
            print(f"  Labels shape: {batch_labels.shape}")
            if i == 1: break
    except Exception as e:
        print(f"Error during validation DataLoader iteration: {e}")
        import traceback
        traceback.print_exc()
            
    print("\nSetup complete. Pixels sampled in collate_fn. Dates are loaded but not part of batch (can be added if needed).")