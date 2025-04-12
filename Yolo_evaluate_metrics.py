from ultralytics import YOLO

def main():
    folder_name = "C:/College/Capstone/YoloModels"
    exp_name = "exp5"
    model_path = f"{folder_name}/{exp_name}/weights/best.pt"
    model = YOLO(model_path)
    class_names = model.names
    data_yaml = "C:/College/Capstone/Filter/Dataset.yaml"

    # --- Evaluation on Validation Set ---
    print("\n--- Validation Set Metrics ---")
    val_metrics = model.val(data=data_yaml, split='val')
    for i, name in class_names.items():
        p, r, ap50, ap = val_metrics.box.class_result(i)
        print(f"\nClass: {name}")
        print(f"  Precision:   {p:.4f}")
        print(f"  Recall:      {r:.4f}")
        print(f"  mAP@0.5:     {ap50:.4f}")
        print(f"  mAP@0.5:0.95:{ap:.4f}")

    # --- Evaluation on Test Set ---
    print("\n--- Test Set Metrics ---")
    test_metrics = model.val(data=data_yaml, split='test')
    for i, name in class_names.items():
        p, r, ap50, ap = test_metrics.box.class_result(i)
        print(f"\nClass: {name}")
        print(f"Precision:   {p:.4f}")
        print(f"Recall:      {r:.4f}")
        print(f"mAP@0.5:     {ap50:.4f}")
        print(f"mAP@0.5:0.95:{ap:.4f}")

if __name__ == "__main__":
    main()
