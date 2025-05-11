import json
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import zarr

# Load labels from the metadata file
with open('timematch_data/denmark/32VNH/2017/meta/labels.json', 'r') as f:
    labels = json.load(f)

# Count occurrences of each category
category_counts = Counter(labels.values())
print(f"Total unique categories: {len(category_counts)}")

# Display the categories and their counts
print("\nCategory distribution:")
for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{category}: {count}")

# Visualize category distribution
plt.figure(figsize=(12, 6))
categories, counts = zip(*sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
plt.bar(categories, counts)
plt.xticks(rotation=90)
plt.title('Category Distribution')
plt.ylabel('Number of examples')
plt.tight_layout()
plt.savefig('category_distribution.png')
plt.show()

# Filter categories with at least 200 examples
valid_categories = [cat for cat, count in category_counts.items() if count >= 200]
print(f"\nCategories with at least 200 examples: {len(valid_categories)}")
for i, category in enumerate(valid_categories):
    print(f"{i}: {category} ({category_counts[category]} examples)")

# Create new indexing for valid categories
category_to_idx = {cat: idx for idx, cat in enumerate(valid_categories)}
idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}

# Count how many parcels remain after filtering
valid_parcels = {parcel_id: cat for parcel_id, cat in labels.items() if cat in valid_categories}
print(f"\nFiltered parcels: {len(valid_parcels)} out of {len(labels)} ({len(valid_parcels)/len(labels)*100:.1f}%)")

# Save the new category indexing
with open('category_mapping.json', 'w') as f:
    json.dump({"category_to_idx": category_to_idx, "idx_to_category": idx_to_category}, f, indent=4)

# Verify data structure by examining one parcel
example_parcel_id = list(valid_parcels.keys())[0]
example_zarr_path = os.path.join('timematch_data/denmark/32VNH/2017/data', f"{example_parcel_id}.zarr")
z = zarr.open(example_zarr_path, mode='r')
print(f"\nExample parcel data shape: {z.shape}")
print(f"This represents: (Time steps: {z.shape[0]}, Channels: {z.shape[1]}, Pixels: {z.shape[2]})")

# Load dates
with open('timematch_data/denmark/32VNH/2017/meta/dates.json', 'r') as f:
    dates = json.load(f)
print(f"\nNumber of dates: {len(dates)}")
print(f"First few dates: {dates[:5]}")

# Convert dates to day-of-year
from datetime import datetime
def date_to_doy(date_str):
    """Convert YYYYMMDD date string to day-of-year (1-365)"""
    date = datetime.strptime(date_str, '%Y%m%d')
    return date.timetuple().tm_yday

dates_doy = [date_to_doy(str(date)) for date in dates]
print(f"\nDates converted to day-of-year: {dates_doy[:5]}...")