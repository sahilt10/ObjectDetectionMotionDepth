import os
import pandas as pd
import shutil

# Define paths
labels_dir = "./labels/train"  # Source labels folder
output_dirs = {
    "val": "./labels/val",
    "test": "./labels/test"
}

# Create destination directories if they don't exist
for folder in output_dirs.values():
    os.makedirs(folder, exist_ok=True)

# Load CSVs
train_csv = pd.read_csv("C:/College/Capstone/Filtered Dataset/train.csv")  # Train images CSV
val_csv = pd.read_csv("C:/College/Capstone/Filtered Dataset/val.csv")      # Val images CSV
test_csv = pd.read_csv("C:/College/Capstone/Filtered Dataset/test.csv")    # Test images CSV

# Function to move label files
def move_labels(csv_file, dest_folder):
    for image_name in csv_file["image_name"]:
        label_file = os.path.splitext(image_name)[0] + ".txt"  # Convert image name to label file name
        src_path = os.path.join(labels_dir, label_file)
        dest_path = os.path.join(dest_folder, label_file)

        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)  # Move the label file

# Move labels based on CSV files
move_labels(val_csv, output_dirs["val"])
move_labels(test_csv, output_dirs["test"])

print("✅ Labels moved successfully!")