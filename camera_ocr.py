"""
camera_ocr.py

A simple real-time OCR tool using OpenCV and Tesseract.

This module captures live video from the default webcam, displays it in a window,
and performs Optical Character Recognition (OCR) when the user presses the 'p' key.
Detected words with high confidence are outlined and labeled in the image.
Press 'q' to exit the application.

Dependencies:
- OpenCV (cv2)
- pytesseract
- Tesseract OCR engine must be installed and accessible in the system path.

"""
import math
import cv2
import pytesseract
import numpy as np
# pylint: disable = no-member


def empty(s):
    """
       Checks if a string is empty or consists only of whitespace.

       Args:
           s (str): The input string.

       Returns:
           bool: True if the string is empty or only whitespace, False otherwise.
       """
    return s.strip() == ""


class CameraOCR:
    """
    A class that captures video from the webcam, allows live preview,
    and performs OCR (Optical Character Recognition) when the 'p' key is pressed.
    Recognized words with sufficient confidence are drawn on the frame.
    """

    def __init__(self):
        """
        Initializes the CameraOCR by creating a video capture object for the default webcam.
        """

        self.cap = cv2.VideoCapture(1)  # 0 is the default webcam

    def run(self):
        """
        Starts the live camera feed.
        - Press 'p' to perform OCR on the current frame and show recognized words.
        - Press 'q' to quit the application and close the window.
        """

        while True:
            ret, frame = self.cap.read()
            # frame = cv2.flip(frame, 1)
            if not ret:
                print("Failed to grab frame.")
                break

            cv2.imshow("Live Feed", frame)
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
                topmost = tuple(
                    largest_contour[largest_contour[:, :, 1].argmin()][0])
                # # red dot at tip
                # cv2.circle(frame, topmost, 10, (0, 0, 255), -1)
                # print("Pen tip at:", topmost)

            if cv2.waitKey(1) & 0xFF == ord('p'):
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                finger_x = topmost[0]
                finger_y = topmost[1]
                min_dist = 1000
                choosen_word = ""
                word_x = 0
                word_y = 0
                data = pytesseract.image_to_data(
                    image_rgb, output_type=pytesseract.Output.DICT)

                n = len(data['text'])
                for i in range(n):
                    if int(data['conf'][i]) > 60 and not empty(data['text'][i]):
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        word = data['text'][i]

                        if (y < finger_y and math.sqrt((finger_x-(x+w/2))**2+(finger_y-(y+h))**2) < min_dist):
                            min_dist = math.sqrt(
                                (finger_x-(x+w/2))**2+(finger_y-y)**2)
                            choosen_word = word
                            word_x = x
                            word_y = y
                        cv2.circle(frame, (int(x+w/2), y+h),
                                   3, (0, 255, 0), -1)
                        cv2.circle(frame, (finger_x, finger_y),
                                   1, (0, 0, 255), -1)
                        cv2.rectangle(frame, (x, y),
                                      (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, word, (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"word x:{word_x} word y:{word_y}")
                print(f"pen x:{finger_x} pen y:{finger_y}")
                print(choosen_word)
                cv2.imshow('capture', frame)
                # cv2.imshow("Blue Pen Mask", mask)
                # print(data)
                # text = pytesseract.image_to_string(image_rgb)
                # print(text)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
