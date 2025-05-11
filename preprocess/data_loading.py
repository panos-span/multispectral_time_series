import json
import os
from collections import Counter
import pandas as pd
import numpy as np
from datetime import datetime

# --- Configuration (You might need to adjust these paths) ---
BASE_DATA_DIR = "timematch_data/denmark/32VNH/2017/" # As per the Greek instructions appendix
LABELS_FILE = os.path.join(BASE_DATA_DIR, "meta", "labels.json")
DATES_FILE = os.path.join(BASE_DATA_DIR, "meta", "dates.json")
MIN_SAMPLES_PER_CLASS = 200

# --- Helper Functions ---
def load_json_file(file_path):
    """Loads a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. File might be corrupted.")
        return None

def get_day_of_year(date_str_list):
    """Converts a list of YYYYMMDD date strings to day of year (1-365/366)."""
    days_of_year = []
    for date_str in date_str_list:
        try:
            # Assuming date_str is in 'YYYYMMDD' format as per paper (page 3 of Greek instructions)
            dt_object = datetime.strptime(date_str, '%Y%m%d')
            # Day of year from 1 to 366
            days_of_year.append(dt_object.timetuple().tm_yday)
        except ValueError:
            print(f"Warning: Could not parse date string {date_str}. Skipping.")
            days_of_year.append(None) # Or handle as an error
    return days_of_year

# --- Simulate Data Loading (Replace with actual loading) ---
# You will need to ensure these files exist at the specified paths or modify the paths.

def simulate_labels_data():
    """Simulates the content of labels.json for demonstration."""
    # Based on TimeMatch paper Fig 6 and common crops
    # Parcel IDs are typically integers or strings
    simulated_labels = {
        "0": "spring_barley", "1": "winter_wheat", "2": "spring_barley",
        "3": "maize", "4": "winter_rapeseed", "5": "winter_wheat",
        # Add more to represent a larger dataset and class imbalance
        **{str(i): "spring_barley" for i in range(6, 250)}, # 244 spring_barley
        **{str(i): "winter_wheat" for i in range(250, 500)}, # 250 winter_wheat
        **{str(i): "maize" for i in range(500, 720)},      # 220 maize
        **{str(i): "winter_rapeseed" for i in range(720, 930)}, # 210 winter_rapeseed
        **{str(i): "sugar_beet" for i in range(930, 1100)}, # 170 sugar_beet (will be filtered)
        **{str(i): "potatoes" for i in range(1100, 1150)}, # 50 potatoes (will be filtered)
        **{str(i): "meadow" for i in range(1150, 1360)},   # 210 meadow
        **{str(i): "oats" for i in range(1360, 1390)},     # 30 oats (will be filtered)
        **{str(i): "rye" for i in range(1390, 1590)},      # 200 rye
        **{str(i): "unknown" for i in range(1590, 1800)}   # 210 unknown (assuming it's a class)
    }
    print(f"Simulating labels data with {len(simulated_labels)} parcels.")
    return simulated_labels

def simulate_dates_data():
    """Simulates the content of dates.json for demonstration."""
    # 52 timesteps as mentioned
    # Dates should be for the year 2017 as per the path
    # Generating some plausible YYYYMMDD date strings for 2017
    # For simplicity, let's make them roughly weekly, but real data will be irregular
    start_date = datetime(2017, 1, 1)
    simulated_dates_raw = []
    for i in range(52):
        # This is a very simplified way to get 52 dates, actual dates will be irregular
        # and from the Sentinel-2 acquisition schedule.
        current_date = start_date + pd.Timedelta(days=i * 7)
        if current_date.year == 2017: # Ensure dates are within 2017
             simulated_dates_raw.append(current_date.strftime('%Y%m%d'))
        else: # if we cross into 2018, wrap around or pick last valid 2017 date
            simulated_dates_raw.append(datetime(2017,12,31).strftime('%Y%m%d'))

    if len(simulated_dates_raw) > 52:
        simulated_dates_raw = simulated_dates_raw[:52]
    elif len(simulated_dates_raw) < 52: # Pad if less than 52
        last_date = simulated_dates_raw[-1] if simulated_dates_raw else datetime(2017,12,31).strftime('%Y%m%d')
        simulated_dates_raw.extend([last_date] * (52 - len(simulated_dates_raw)))

    print(f"Simulating dates data with {len(simulated_dates_raw)} acquisition dates.")
    return simulated_dates_raw

# --- Main Preprocessing Steps ---

# 1. Load Labels
# parcel_labels = load_json_file(LABELS_FILE) # Use this for actual file
parcel_labels = simulate_labels_data() # Using simulated data for now

if parcel_labels is None:
    print("Could not load parcel labels. Exiting.")
    exit()

# Convert to a list of (parcel_id, label_name) tuples for easier processing
parcel_label_list = list(parcel_labels.items())
labels_df = pd.DataFrame(parcel_label_list, columns=['parcel_id', 'crop_type'])
print(f"\nLoaded {len(labels_df)} parcel labels into DataFrame.")
print("Initial class distribution:")
print(labels_df['crop_type'].value_counts())

# 2. Filter Categories
print(f"\nFiltering categories with fewer than {MIN_SAMPLES_PER_CLASS} samples...")
class_counts = labels_df['crop_type'].value_counts()
classes_to_keep = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()

# Ensure 'unknown' is handled if present, usually kept or mapped specifically
# The TimeMatch paper mentions an "unknown" class (page 4, and Figure 6 on page 9)
# If 'unknown' is in classes_to_keep, it's fine. If not, but desired, add it.
# For now, we'll just keep classes meeting the threshold.

filtered_labels_df = labels_df[labels_df['crop_type'].isin(classes_to_keep)].copy() # Use .copy() to avoid SettingWithCopyWarning

print(f"\nClasses kept after filtering (>= {MIN_SAMPLES_PER_CLASS} samples): {classes_to_keep}")
print(f"Number of parcels after filtering: {len(filtered_labels_df)}")
print("Class distribution after filtering:")
print(filtered_labels_df['crop_type'].value_counts())

if filtered_labels_df.empty:
    print("No classes met the filtering criteria. Exiting or check MIN_SAMPLES_PER_CLASS.")
    exit()

# 3. Create New Indexing for Categories
# Sort classes for consistent mapping
sorted_classes_to_keep = sorted(list(filtered_labels_df['crop_type'].unique()))

class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted_classes_to_keep)}
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

filtered_labels_df['class_idx'] = filtered_labels_df['crop_type'].map(class_to_idx)

print("\nNew class indexing:")
print("Class to Index mapping:", class_to_idx)
print("Index to Class mapping:", idx_to_class)
print("\nDataFrame with new class indices (first 5 rows):")
print(filtered_labels_df.head())

# 4. Load and Process Dates
# acquisition_dates_raw = load_json_file(DATES_FILE) # Use this for actual file
acquisition_dates_raw = simulate_dates_data() # Using simulated data

if acquisition_dates_raw is None:
    print("Could not load acquisition dates. Some features might not be available.")
    acquisition_doy = None
else:
    if not isinstance(acquisition_dates_raw, list):
        # The paper appendix (page 3 of Greek) says dates.json "αναφέρει τις ημερομηνίες ... σε μορφή ΕΕΕΕΜΜΗΗ"
        # This suggests a list of dates. If it's a dict, adjust accordingly.
        # For now, assuming it's a list of 52 date strings.
        print(f"Warning: Expected dates.json to contain a list of date strings. Type found: {type(acquisition_dates_raw)}")
        acquisition_doy = None
    else:
        acquisition_doy = get_day_of_year(acquisition_dates_raw)
        if acquisition_doy and len(acquisition_doy) == 52:
            print(f"\nSuccessfully processed {len(acquisition_doy)} acquisition dates into days of year (first 5):")
            print(acquisition_doy[:5])
        else:
            print(f"\nWarning: Expected 52 dates, but got {len(acquisition_doy) if acquisition_doy else 0}. Check dates.json.")
            if acquisition_doy: print(acquisition_doy)


# At this point, filtered_labels_df contains the parcels to be used,
# with their original crop type and new integer class index.
# acquisition_doy contains the day of year for each of the 52 timesteps.

# --- Next Steps (Data Loading for SITS) ---
# The next major step would be to load the actual Satellite Image Time Series (SITS)
# data for the parcels in `filtered_labels_df` from the .zarr files.
# This involves:
# - Iterating through `filtered_labels_df['parcel_id']`.
# - Locating the corresponding .zarr file and the data within it.
# - The Greek instructions mention `timematch_data/denmark/32VNH/2017/data/{0..N}.zarr`
#   and then further indexing within those zarr files, likely by parcel_id.
#   Need to confirm how parcel_ids from labels.json map to zarr file structure.
#   The appendix shows `timematch_data/denmark/32VNH/2017/data/{0..4}.zarr` and `meta/parcels`
#   This implies parcels might be grouped into larger zarr files, and `meta/parcels` or `metadata.pkl`
#   might contain the mapping from a global parcel_id to its location within a specific zarr file.
#   For now, we will assume we can retrieve the (52, num_pixels, 10_bands) data for each parcel.

print("\nPreprocessing steps 1 & 2 (category analysis and filtering) complete.")
print("Next, you would proceed to load the SITS data for the filtered parcels and implement the dataloaders.")

# Store the results for later use
# For example, you might want to save the filtered_labels_df and class_to_idx
# filtered_labels_df.to_csv("filtered_parcel_list.csv", index=False)
# with open("class_map.json", 'w') as f:
# json.dump(class_to_idx, f)

# To use in subsequent steps:
# final_parcel_ids = filtered_labels_df['parcel_id'].tolist()
# final_parcel_labels = filtered_labels_df['class_idx'].tolist()
# num_classes = len(class_to_idx)