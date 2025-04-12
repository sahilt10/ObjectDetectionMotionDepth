from pycocotools.coco import COCO
import os

coco_json = "C:/College/Capstone/ObjectDetectionMotionDepth/updated_instances_train2017.json"  # Your updated COCO JSON
output_dir = "labels"  # Change to "labels/val/" for validation
os.makedirs(output_dir, exist_ok=True)

# Animal -> 0, Vehicle -> 1, Person -> 2
valid_classes = {0, 1, 2}

# Loading COCO
coco = COCO(coco_json)

# Converting to YOLO format
for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    label_lines = []
    for ann in anns:
        cat_id = ann["category_id"]
        if cat_id not in valid_classes:
            print(f"⚠️ Skipping unknown category_id: {cat_id}")
            continue

        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_info["width"]
        y_center = (y + h / 2) / img_info["height"]
        w_norm = w / img_info["width"]
        h_norm = h / img_info["height"]

        label_lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    if label_lines:
        label_path = os.path.join(output_dir, img_info["file_name"].replace('.jpg', '.txt'))
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))
