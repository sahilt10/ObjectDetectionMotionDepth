import os
import pandas as pd
import shutil

labels_dir = "./labels"
output_dirs = {
    "val": "C:/College/Capstone/Filter/labels/val",
    "test": "C:/College/Capstone/Filter/labels/test",
    "train": "C:/College/Capstone/Filter/labels/train"
}

# Creating destination directories if they don't exist
for folder in output_dirs.values():
    os.makedirs(folder, exist_ok=True)

# Loading CSVs
train_csv = pd.read_csv("C:/College/Capstone/Filter/train.csv") 
val_csv = pd.read_csv("C:/College/Capstone/Filter/val.csv")      
test_csv = pd.read_csv("C:/College/Capstone/Filter/test.csv")    

# Function to move label files
def move_labels(csv_file, dest_folder):
    for image_name in csv_file["image_name"]:
        label_file = os.path.splitext(image_name)[0] + ".txt" 
        src_path = os.path.join(labels_dir, label_file)
        dest_path = os.path.join(dest_folder, label_file)

        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)

move_labels(val_csv, output_dirs["val"])
move_labels(test_csv, output_dirs["test"])
move_labels(train_csv, output_dirs["train"])

print("✅ Labels moved successfully!")