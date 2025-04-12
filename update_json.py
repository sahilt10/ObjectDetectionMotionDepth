import json

# Define grouped class mappings
CLASS_MAP = {
    1: 2,  # Person -> class 2
    2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,  # Vehicles -> class 1
    16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0  # Animals -> class0
}

# Load COCO annotations
with open("C:/College/Capstone/Dataset/annotations_trainval2017/instances_train2017.json") as f:
    coco_data = json.load(f)

# Filter and update annotations
updated_annotations = []
skipped = 0
for ann in coco_data["annotations"]:
    old_id = ann["category_id"]
    if old_id in CLASS_MAP:
        ann["category_id"] = CLASS_MAP[old_id]
        updated_annotations.append(ann)
    else:
        skipped += 1  # Optional: count skipped annotations

# Update categories list to only 3 grouped classes
new_categories = [
    {"id": 1, "name": "vehicle"},
    {"id": 2, "name": "person"},
    {"id": 0, "name": "animal"}
]

# Update the JSON
coco_data["annotations"] = updated_annotations
coco_data["categories"] = new_categories

# Save updated JSON
with open("C:/College/Capstone/ObjectDetectionMotionDepth/updated_instances_train2017.json", "w") as f:
    json.dump(coco_data, f)

print("✅ JSON updated and saved.")
print(f"🗑️ Skipped annotations (not in CLASS_MAP): {skipped}")