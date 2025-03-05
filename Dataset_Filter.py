import os
import shutil
import random
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt").to("cuda")  # You can use 'yolov8s.pt' for better accuracy

# Define relevant object classes (from COCO dataset)
RELEVANT_CLASSES = {"person", "car", "bus", "truck", "motorcycle", "bicycle", "horse", "dog", "cat", "bird", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

def detect_objects_batch(image_paths):
    """Runs object detection on a batch of images and returns a set of valid image names."""
    valid_names = set()
    
    # Run YOLO inference on a batch of images
    results = model(image_paths, device="cuda" if model.device.type != "cpu" else "cpu")  # Use GPU if available
    
    for img_path, result in zip(image_paths, results):
        detected_classes = {model.names[int(cls)] for cls in result.boxes.cls}  # Get detected class names
        if detected_classes & RELEVANT_CLASSES:  # Check if any relevant class is detected
            valid_names.add(os.path.splitext(os.path.basename(img_path))[0])  # Store name without extension
    
    return valid_names


def create_subset(original_dataset_path, subset_dataset_path, detection_results, num_samples=1000):
    train_images_path = os.path.join(original_dataset_path, "train2017")
    train_masks_path = os.path.join(original_dataset_path, "panoptic_annotations_trainval2017/panoptic_train2017")
    
    subset_train_images_path = os.path.join(subset_dataset_path, "train images")
    subset_train_masks_path = os.path.join(subset_dataset_path, "annotations")
    
    os.makedirs(subset_train_images_path, exist_ok=True)
    os.makedirs(subset_train_masks_path, exist_ok=True)
    
    image_files = sorted(os.listdir(train_images_path))
    mask_files = sorted(os.listdir(train_masks_path))

    print(f"Total train images found: {len(image_files)}")
    print(f"Total train masks found: {len(mask_files)}")

    # Remove file extensions for proper matching
    image_names = {os.path.splitext(f)[0] for f in image_files}
    mask_names = {os.path.splitext(f)[0] for f in mask_files}
    
    # Find common names (images that have corresponding masks)
    valid_names = list(image_names & mask_names)

    # Filter images that contain relevant objects using batch inference
    relevant_images = set()
    for i in range(0, len(valid_names), 32):
        batch_files = [os.path.join(train_images_path, name + ".jpg") for name in valid_names[i:i+32]]
        relevant_images.update(detect_objects_batch(batch_files))

    filtered_names = list(relevant_images)

    print(f"Total valid matching pairs (vehicles, animals, persons): {len(filtered_names)}")

    actual_sample_size = min(len(filtered_names), num_samples)

    if actual_sample_size == 0:
        print("Error: No matching image-mask pairs found with relevant objects!")
        return

    if actual_sample_size < num_samples:
        print(f"Warning: Only {actual_sample_size} valid pairs found, selecting all available.")

    selected_names = random.sample(filtered_names, actual_sample_size)

    for name in selected_names:
        # Find the correct extensions
        img_file = next(f for f in image_files if os.path.splitext(f)[0] == name)
        mask_file = next(f for f in mask_files if os.path.splitext(f)[0] == name)

        shutil.copy(os.path.join(train_images_path, img_file), os.path.join(subset_train_images_path, img_file))
        shutil.copy(os.path.join(train_masks_path, mask_file), os.path.join(subset_train_masks_path, mask_file))
    
    print(f"Subset created with {actual_sample_size} images and masks in {subset_dataset_path}")


# Example usage
original_dataset = "C:/College/Capstone/Dataset"  # Change this
target_dataset = "C:/College/Capstone/Sub-Dataset"  # Change this
num_samples = 1000

create_subset(original_dataset, target_dataset, num_samples)