import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture('data/videos/vid1.mp4')

# Initialize the background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor to the frame
    fg_mask = back_sub.apply(frame)

    # Apply a binary threshold to clean up the mask
    _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Morphological operations to remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Find contours to identify large moving objects (cars)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new mask to draw detected cars
    mask = np.zeros_like(binary)

    for contour in contours:
        # Filter out small contours based on area
        if cv2.contourArea(contour) > 500:  # Adjust this threshold as needed
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Display the processed video
    cv2.imshow('Cars Highlighted', mask)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

