import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Define dataset paths
csv_path = "C:/College/Capstone/Filtered Dataset/filtered_dataset.csv"  # Path to the CSV file
image_dir = "C:/College/Capstone/Filtered Dataset"  # Path to original filtered images
output_dir = "C:/College/Capstone/Filtered Dataset"  # Root directory for train, test, val
os.makedirs(output_dir, exist_ok=True)

# Create train, test, val folders
for folder in ["train", "test", "val"]:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

df["none"] = ((df["persons"] == 0) & (df["animals"] == 0) & (df["vehicles"] == 0)).astype(int)

# Stratified split based on category counts
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Function to check category distribution
def get_category_counts(df):
    return {
        "persons": df["persons"].sum(),
        "animals": df["animals"].sum(),
        "vehicles": df["vehicles"].sum(),
        "none": df["none"].sum()
    }

# Get category distributions for each split
train_counts = get_category_counts(train_df)
test_counts = get_category_counts(test_df)
val_counts = get_category_counts(val_df)

# Print category distributions
print("Category Distribution:")
print(f"✅ Train: {train_counts}")
print(f"✅ Test: {test_counts}")
print(f"✅ Val: {val_counts}")

# Function to move images to respective folders
def move_images(image_list, folder_name):
    folder_path = os.path.join(output_dir, folder_name)
    for image_name in image_list:
        src_path = os.path.join(image_dir, image_name)
        dst_path = os.path.join(folder_path, image_name)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)  # Move instead of copy

# Move images into train, test, val folders
move_images(train_df["image_name"], "train")
move_images(test_df["image_name"], "test")
move_images(val_df["image_name"], "val")

# Save CSVs for each split
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)

# Print results
print(f"✅ Train: {len(train_df)} images moved")
print(f"✅ Test: {len(test_df)} images moved")
print(f"✅ Val: {len(val_df)} images moved")
print(f"Dataset split and images moved to {output_dir}")