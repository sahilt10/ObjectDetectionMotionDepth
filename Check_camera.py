import cv2

# Opening both cameras
cam1 = cv2.VideoCapture(2)  
cam2 = cv2.VideoCapture(1)  

while True:
    # Reading frames from both cameras
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        print("Error: Could not read from one or both cameras.")
        break

    # Resizing for better visualization
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))

    # Stacking images side by side
    combined = cv2.hconcat([frame1, frame2])

    # Showing both camera feeds
    cv2.imshow("Stereo Camera Feed", combined)

    # Exiting on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing resources
cam1.release()
cam2.release()
cv2.destroyAllWindows()
