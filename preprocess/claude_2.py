import json
import os
import pickle
from collections import Counter

# --- Configuration ---
# Adjust this path to where your 'timematch_data' folder is located
BASE_DATA_PATH = "./timematch_data" # Or the full path if it's elsewhere
REGION = "denmark"
TILE = "32VNH"
YEAR = "2017"

# Construct paths based on the configuration
# Path to the specific tile's metadata
METADATA_PATH = os.path.join(BASE_DATA_PATH, REGION, TILE, YEAR, "meta")
LABELS_FILE = os.path.join(METADATA_PATH, "labels.json")
METADATA_PKL_FILE = os.path.join(METADATA_PATH, "metadata.pkl") # Contains parcel IDs

# --- Helper Functions ---
def load_json(file_path):
    """Loads a JSON file."""
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
    """Loads a pickle file."""
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

# --- Main Script ---
if __name__ == "__main__":
    print(f"Processing dataset for: {REGION} - {TILE} - {YEAR}")
    print("-" * 30)

    # 1. Load labels data
    # The keys in labels.json are parcel IDs (as strings), values are crop names
    labels_data = load_json(LABELS_FILE)
    if labels_data is None:
        print("Exiting due to error loading labels.json.")
        exit()

    # (Optional but good practice) Load parcel IDs from metadata.pkl
    # This helps verify we are considering all parcels defined in the subset
    metadata_content = load_pickle(METADATA_PKL_FILE)
    if metadata_content is None:
        print("Warning: Could not load metadata.pkl. Proceeding with labels.json keys only.")
        parcel_ids_from_meta = list(labels_data.keys()) # Fallback
    else:
        # Assuming 'parcels' key in metadata.pkl contains the list of parcel IDs (often as integers)
        # The keys in labels.json are strings, so we need to ensure consistency
        if 'parcels' in metadata_content:
            parcel_ids_from_meta = [str(p_id) for p_id in metadata_content['parcels']]
            print(f"Found {len(parcel_ids_from_meta)} parcel IDs in metadata.pkl.")
            # Check if all parcel_ids from metadata are in labels_data
            missing_in_labels = [p_id for p_id in parcel_ids_from_meta if p_id not in labels_data]
            if missing_in_labels:
                print(f"Warning: {len(missing_in_labels)} parcel IDs from metadata.pkl are not in labels.json.")
                # For this exercise, we will only use parcels present in labels.json
        else:
            print("Warning: 'parcels' key not found in metadata.pkl. Using labels.json keys as parcel IDs.")
            parcel_ids_from_meta = list(labels_data.keys()) # Fallback

    # We will work with parcel IDs that are present in labels_data
    # This implicitly means we only consider parcels for which we have labels.
    all_crop_labels = [labels_data[parcel_id] for parcel_id in labels_data.keys()]
    
    print("\n--- 1. Original Crop Categories and Counts ---")
    crop_counts = Counter(all_crop_labels)
    print(f"Found {len(crop_counts)} unique crop categories.")
    for crop, count in crop_counts.most_common():
        print(f"- {crop}: {count} parcels")

    # 2. Filter categories with fewer than 200 examples
    print("\n--- 2. Filtering Categories (Min 200 Examples) ---")
    MIN_EXAMPLES_THRESHOLD = 200
    
    crop_types_to_keep = []
    crop_types_to_remove = []

    for crop, count in crop_counts.items():
        if count >= MIN_EXAMPLES_THRESHOLD:
            crop_types_to_keep.append(crop)
        else:
            crop_types_to_remove.append(crop)
            
    print(f"Keeping {len(crop_types_to_keep)} categories with >= {MIN_EXAMPLES_THRESHOLD} examples:")
    for crop in sorted(crop_types_to_keep): # Sort for consistent output
        print(f"- {crop} (Count: {crop_counts[crop]})")
        
    if crop_types_to_remove:
        print(f"\nRemoving {len(crop_types_to_remove)} categories with < {MIN_EXAMPLES_THRESHOLD} examples:")
        for crop in sorted(crop_types_to_remove):
             print(f"- {crop} (Count: {crop_counts[crop]})")
    else:
        print("\nNo categories to remove based on the threshold.")

    # Create a new dictionary of labels containing only the filtered crop types
    # And a list of parcel_ids that are kept
    filtered_labels_data = {}
    kept_parcel_ids = []

    for parcel_id, crop_name in labels_data.items():
        if crop_name in crop_types_to_keep:
            filtered_labels_data[parcel_id] = crop_name
            kept_parcel_ids.append(parcel_id)
            
    print(f"\nNumber of parcels after filtering: {len(filtered_labels_data)}")
    print(f"Number of unique crop types after filtering: {len(set(filtered_labels_data.values()))}")

    # 3. Define new indexing for the remaining categories
    print("\n--- 3. New Indexing for Kept Categories ---")
    # Sort the kept crop types alphabetically for consistent indexing
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

    # --- Storing results for later use (optional, but good practice) ---
    # These variables will be useful for the next steps of data loading and model training:
    # - kept_parcel_ids: List of parcel IDs to use for training/validation/testing
    # - filtered_labels_data: Dictionary mapping kept_parcel_id -> crop_name
    # - crop_to_idx: Mapping from crop_name to integer label
    # - idx_to_crop: Mapping from integer label to crop_name
    # - num_classes: Number of unique classes
    
    print("\n--- Summary of Data for Next Steps ---")
    print(f"Number of parcels to be used: {len(kept_parcel_ids)}")
    print(f"Number of classes: {num_classes}")
    print("Ready for data loading and model building with these filtered and indexed classes.")