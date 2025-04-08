import cv2

# Open both cameras (change index if needed)
cam1 = cv2.VideoCapture(2)  # First camera
cam2 = cv2.VideoCapture(1)  # Second camera

while True:
    # Read frames from both cameras
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        print("Error: Could not read from one or both cameras.")
        break

    # Resize for better visualization
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))

    # Stack images side by side
    combined = cv2.hconcat([frame1, frame2])

    # Show both camera feeds
    cv2.imshow("Stereo Camera Feed", combined)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam1.release()
cam2.release()
cv2.destroyAllWindows()
