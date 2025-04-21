import cv2
import numpy as np
# pylint: disable = no-member
cap = cv2.VideoCapture(1)  # Open the webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for blue
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask where blue is white and everything else is black
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        cv2.circle(frame, topmost, 10, (0, 0, 255), -1)  # red dot at tip
        print("Pen tip at:", topmost)
    # Optional: show both original frame and mask side by side
    cv2.imshow("Webcam", frame)
    cv2.imshow("Blue Pen Mask", mask)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
