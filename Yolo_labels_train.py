from pycocotools.coco import COCO
import os

# Paths
coco_json = "C:/College/Capstone/ObjectDetectionMotionDepth/updated_instances_train2017.json"  # Your updated COCO JSON
output_dir = "labels/train/"  # Change to "labels/val/" for validation
os.makedirs(output_dir, exist_ok=True)

# Load COCO JSON
coco = COCO(coco_json)

# Map COCO categories to YOLO class IDs
category_map = {
    1: 0,  # Person -> 0
    2: 1,  # Vehicle -> 1
    16: 2, # Animal -> 2
    # Add other category mappings here
}

# Convert to YOLO format
for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    label_path = os.path.join(output_dir, f"{img_info['file_name'].replace('.jpg', '.txt')}")
    with open(label_path, "w") as f:
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id in category_map:
                x, y, w, h = ann["bbox"]
                x_center = x + w / 2
                y_center = y + h / 2
                f.write(f"{category_map[cat_id]} {x_center/img_info['width']} {y_center/img_info['height']} {w/img_info['width']} {h/img_info['height']}\n")