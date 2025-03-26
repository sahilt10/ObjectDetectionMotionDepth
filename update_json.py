import json

# Define grouped class mappings
CLASS_MAP = {
    1: 2,  # Person -> class 2
    2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,  # Vehicles -> class 1
    16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0  # Animals -> class 0
}

# Load COCO annotations
with open("C:/College/Capstone/Dataset/annotations_trainval2017/instances_train2017.json") as f:
    coco_data = json.load(f)

# Update category IDs to new grouped classes
for annotation in coco_data["annotations"]:
    category_id = annotation["category_id"]
    if category_id in CLASS_MAP:
        annotation["category_id"] = CLASS_MAP[category_id]  # Replace with grouped class ID

# Save updated annotations
with open("updated_instances_train2017.json", "w") as f:
    json.dump(coco_data, f)