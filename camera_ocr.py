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

import cv2
import pytesseract
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
        self.cap = cv2.VideoCapture(0)  # 0 is the default webcam

    def run(self):
        """
        Starts the live camera feed.
        - Press 'p' to perform OCR on the current frame and show recognized words.
        - Press 'q' to quit the application and close the window.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            cv2.imshow("Live Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('p'):
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                        cv2.rectangle(image_rgb, (x, y),
                                      (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(image_rgb, word, (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.imshow('capture', image_rgb)

                print(data)
                # text = pytesseract.image_to_string(image_rgb)
                # print(text)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
