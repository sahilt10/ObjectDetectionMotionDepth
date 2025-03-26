import json
import pandas as pd
import os
import shutil
import random

# Define paths (update these)
json_path = "C:/College/Capstone/Dataset/annotations_trainval2017/instances_train2017.json" # Path to COCO annotations
image_dir = "C:/College/Capstone/Dataset/train2017/"  # Folder containing COCO train images
output_dataset_dir = "C:/College/Capstone/Filtered Dataset"  # New folder to store 7000 selected images
output_csv_path = "C:/College/Capstone/Filtered Dataset/filtered_dataset.csv"  # CSV file path

# Define COCO category IDs
ANIMAL_CATEGORIES = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25}  # Animals
VEHICLE_CATEGORIES = {2, 3, 4, 5, 6, 7}  # Vehicles
PERSON_CATEGORY = {1}  # Persons

# Load COCO annotations
with open(json_path, "r") as f:
    coco_data = json.load(f)

# Create output dataset directory if it doesn't exist
os.makedirs(output_dataset_dir, exist_ok=True)

# Dictionary to store category counts per image
image_categories = {}

# Step 1: Process annotations to count categories per image
for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    category_id = ann["category_id"]

    if image_id not in image_categories:
        image_categories[image_id] = {"animals": 0, "vehicles": 0, "persons": 0}

    if category_id in ANIMAL_CATEGORIES:
        image_categories[image_id]["animals"] += 1
    elif category_id in VEHICLE_CATEGORIES:
        image_categories[image_id]["vehicles"] += 1
    elif category_id in PERSON_CATEGORY:
        image_categories[image_id]["persons"] += 1

# Step 2: Collect images
selected_images = []
no_category_images = []  # Images with neither animals, vehicles, nor persons

for img in coco_data["images"]:
    image_id = img["id"]
    image_name = img["file_name"]
    
    # Default category counts
    animals = image_categories.get(image_id, {}).get("animals", 0)
    vehicles = image_categories.get(image_id, {}).get("vehicles", 0)
    persons = image_categories.get(image_id, {}).get("persons", 0)

    # Identify images with no relevant categories
    if animals == 0 and vehicles == 0 and persons == 0:
        no_category_images.append((image_name, animals, vehicles, persons))
    else:
        selected_images.append((image_name, animals, vehicles, persons))

# Step 3: Randomly pick 7000 images (including some with no categories)
random.shuffle(selected_images)  # Shuffle for randomness
random.shuffle(no_category_images)  # Shuffle the images with no relevant categories

num_no_category = max(500, len(no_category_images) // 10)  # Ensure a small portion has no category
final_images = selected_images[:(10000 - num_no_category)] + no_category_images[:num_no_category]

# Step 4: Copy selected images to new dataset folder
dataset_records = []
for image_name, animals, vehicles, persons in final_images:
    src_path = os.path.join(image_dir, image_name)
    dst_path = os.path.join(output_dataset_dir, image_name)

    if os.path.exists(src_path):  # Ensure the image exists before copying
        shutil.copy(src_path, dst_path)

        # Append to dataset records
        dataset_records.append({
            "image_name": image_name,
            "animals": animals,
            "vehicles": vehicles,
            "persons": persons
        })

# Step 5: Save results to CSV
df = pd.DataFrame(dataset_records)
df.to_csv(output_csv_path, index=False)

# Print summary
print(f"✅ Dataset saved! {len(final_images)} images stored in: {output_dataset_dir}")
print(f"✅ CSV file saved at: {output_csv_path}")