from ultralytics import YOLO

# Load the YOLO model with pre-trained weights
model = YOLO("yolov8n.pt")  # Load YOLOv8 nano pre-trained weights
folder_name = "C:/College/Capstone/YoloModels" 
exp_name="exp1"

# Freeze first 10 layers manually
for param in list(model.model.parameters())[:10]:
    param.requires_grad = False  # Freeze layers

# Train only the final layers
model.train(
    data="C:/College/Capstone/Filtered Dataset/Dataset.yaml",  # Path to dataset.yaml
    epochs=100,  # Increase for better accuracy
    batch=16,  # Adjust based on GPU memory
    imgsz=640,  # Image size
    lr0=0.0005,  # Learning rate (lower for fine-tuning)
    device="cuda",  # Use GPU
    workers=0,  # Speed up training
    project=folder_name,  # Custom folder for saving results
    name=exp_name,  # Experiment name
    exist_ok=False
)

# Load the fine-tuned model
model = YOLO(f"{folder_name}/{exp_name}/weights/best.pt")

# Evaluate on Train set
train_metrics = model.val(split="train", workers=0)
print(f"Train mAP@50: {train_metrics.box.map:.4f}")  # Train Mean Average Precision
print(f"Train Precision: {train_metrics.box.mp:.4f}")
print(f"Train Recall: {train_metrics.box.mr:.4f}")

# Evaluate on Validation set
val_metrics = model.val(split="val", workers=0)
print(f"Validation mAP@50: {val_metrics.box.map:.4f}")  # Validation Mean Average Precision
print(f"Validation Precision: {val_metrics.box.mp:.4f}")
print(f"Validation Recall: {val_metrics.box.mr:.4f}")

# Evaluate on Test set
test_metrics = model.val(split="test", workers=0)
print(f"Test mAP@50: {test_metrics.box.map:.4f}")  # Test Mean Average Precision
print(f"Test Precision: {test_metrics.box.mp:.4f}")
print(f"Test Recall: {test_metrics.box.mr:.4f}")