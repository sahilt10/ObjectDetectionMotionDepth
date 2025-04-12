from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.yaml").load("yolov8n.pt") 
    folder_name = "C:/College/Capstone/YoloModels"
    exp_name = "exp5"

    #Freezing 5 layers
    for param in list(model.model.parameters())[:5]:
        param.requires_grad = False

    model.train(
        data="C:/College/Capstone/Filter/Dataset.yaml",
        epochs=100,
        imgsz=704,
        batch=16,
        workers=2,
        verbose=True,
        plots=True,
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.2,
        translate=0.2,
        scale=0.5,
        shear=0.2,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1,
        mixup=0.075,
        copy_paste=0.15,
        close_mosaic=10,
        amp=True,
        cache='disk',
        resume=False,
        save_period=5,  
        project=folder_name,
        name=exp_name,
        exist_ok=False
    )

if __name__ == "__main__":
    main()