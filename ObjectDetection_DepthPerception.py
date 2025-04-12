import cv2
import numpy as np
from filterpy.kalman import KalmanFilter  # Ensure filter.py exists
from ultralytics import YOLO  # Load YOLO model

# Defining your folder and experiment name
folder_name = "C:/College/Capstone/YoloModels" 
exp_name="exp1"

# Loading fine-tuned YOLO model
model = YOLO(f"{folder_name}/{exp_name}/weights/best.pt")

# Kalman Filter initialization
kf = KalmanFilter(dim_x=4, dim_z=2)  

# Stereo camera parameters
focal_length = 640  
baseline = 0.275  

# Opening video capture for stereo cameras
cap_left = cv2.VideoCapture(2)   
cap_right = cv2.VideoCapture(1)  

def calculate_depth(disparity):
    """ Compute depth from disparity using stereo camera parameters """
    if disparity > 0:
        return (focal_length * baseline) / disparity
    return 0  

while cap_left.isOpened() and cap_right.isOpened():
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    
    if not retL or not retR:
        print("Failed to capture frames")
        break

    # Performing YOLO object detection
    results = model(frameL)  

    for result in results:  
        boxes = result.boxes  

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  #
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class label

            # Computing center of bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Computing disparity 
            disparity = abs(frameL[center_y, center_x, 0] - frameR[center_y, center_x, 0])
            depth = calculate_depth(disparity)

            # Applying Kalman Filter for smoothing
            measurement = np.array([[center_x], [center_y]], dtype=np.float32)
            kf.predict()
            kf.update(measurement)
            filtered_x, filtered_y = kf.x[:2]

            # Drawing bounding box and depth info
            cv2.rectangle(frameL, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frameL, f"Depth: {depth:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Showing output video
    cv2.imshow("Object Detection with Depth", frameL)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
