import cv2
import numpy as np

# Camera parameters
F = 640 # Focal length in pixels
B = 0.275  # distance between cameras in meters

# Open two cameras 
cap_left = cv2.VideoCapture(2)
cap_right = cv2.VideoCapture(1)

# StereoSGBM matcher (for computing disparity)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*5,  
    blockSize=15,
    P1=8 * 3 * 15**2,
    P2=32 * 3 * 15**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

def mouse_callback(event, x, y, flags, param):
    """ Callback function to display absolute depth on mouse click """
    if event == cv2.EVENT_LBUTTONDOWN:
        disparity_value = disparity[y, x]
        if disparity_value > 0:  # Avoid division by zero
            depth = (F * B) / disparity_value
            print(f"Depth at ({x}, {y}): {depth:.2f} meters")
        else:
            print("Invalid disparity value at this point.")

while True:
    # Reading frames from both cameras
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Error: Couldn't capture images.")
        break

    # Converting frames to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Computing disparity map
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Normalizing for better visualization
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # Displaying results
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Disparity Map", disp_vis)

    # Setting mouse callback for depth calculation
    cv2.setMouseCallback("Disparity Map", mouse_callback)

    # Exiting on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing resources
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
