"""
camera_ocr.py

A simple real-time OCR tool using OpenCV and Tesseract.

This module captures live video from the default webcam, displays it in a window,
recognize the tip of a pen and performs Optical Character Recognition (OCR) when the user
put the pen under a word long enough. it then translate the word to english and read it out loud.
Press 'q' to exit the application.

Dependencies:
- OpenCV (cv2)
- pytesseract
- Tesseract OCR engine must be installed and accessible in the system path.

"""
import math
import time
import os
import requests
import cv2
import pytesseract
import numpy as np
import pyttsx3
from dotenv import load_dotenv

# pylint: disable = no-member
load_dotenv()


def dist(p1: tuple, p2: tuple) -> float:
    """
    Calculate the Euclidean distance between two 2D points.

    Args:
        p1: A tuple (x, y) representing the first point.
        p2: A tuple (x, y) representing the second point.

    Returns:
        The distance as a float.
    """
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def translate(word: str) -> str:
    """
    Translate a French word into English using Google Translate API.

    Args:
        word: A string containing a single French word to translate.

    Returns:
        The translated English word as a string.

    Raises:
        ValueError: If the API key is not set.
        requests.RequestException: If the HTTP request fails.
    """
    api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing environment variable: GOOGLE_TRANSLATE_API_KEY")
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        "key": api_key,
        "source": "fr",
        "target": "en",
        "q": word
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()  # Raise error for bad HTTP codes
        data = response.json()
        translated_text = data["data"]["translations"][0]["translatedText"]
        return translated_text
    except requests.RequestException as e:
        print("Error:", e)
        raise


def word_to_speak(word: str) -> None:
    """
    Gets a word as a string and reads it out loud.
    """
    engine = pyttsx3.init()
    # Set a common English voice directly (for Windows)
    engine.setProperty(
        'voice',
        'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0')
    engine.say(word)
    engine.runAndWait()


def empty(s: str) -> bool:
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
    A class that captures video from the webcam, allows live preview, recognize a pen
    and performs OCR (Optical Character Recognition) when the pen stays long enough under a word.
    It then translate the word and reads it out loud.
    """

    def __init__(self, source=1, output_file=None):
        """
        Initializes the CameraOCR by creating a video capture object for the default webcam.
        """
        self.cap = cv2.VideoCapture(source)  # 1 is my phone cam
        self.output_file = output_file
        self.test_mode = output_file is not None  # return true if test mode is on

    def run(self):
        """
        Starts the live camera feed.
        - Press 'q' to quit the application and close the window.
        """
        distance_threshold = 3  # How many pixels considered a move of the pen
        time_still_treshold = 1  # How long (in secs) the pen needs to stay
        time_since_last_move = time.time()
        time_still = 0
        last_location = (0, 0)
        pen_location = (0, 0)  # topmost point in the pen
        translated = True  # signify that the current word was translated already
        delay = 1
        if self.test_mode:
            print("Processing video...")

        while True:
            print(f"pen location:{pen_location}")
            ret, frame = self.cap.read()
            if not ret:
                print("Video ended or failed to grab frame.")
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
            min_area = 10  # define min area to prefent false positive of noise

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) >= min_area:
                    pen_location = tuple(
                        largest_contour[largest_contour[:, :, 1].argmin()][0])
                else:
                    pen_location = (0, 0)
            time_still = time.time()-time_since_last_move
            # Check how long the pen stays in the same spot
            if dist(pen_location, last_location) > distance_threshold:
                time_since_last_move = time.time()
                last_location = pen_location
                translated = False
            elif pen_location != (0, 0) and not translated and time_still > time_still_treshold:
                choosen_word = ""
                min_dist = 1000
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
                        # middle buttom of the word
                        word_loc = (int(x+w/2), y+h)
                        if y < pen_location[1] and dist(pen_location, word_loc) < min_dist:
                            min_dist = dist(pen_location, word_loc)
                            choosen_word = word
                        # temporary

                        cv2.circle(frame, word_loc, 3, (0, 255, 0), -1)
                        cv2.circle(frame, pen_location, 1, (0, 0, 255), -1)
                        cv2.rectangle(frame, (x, y), (x+w, y+h),
                                      (255, 0, 0), 2)
                        cv2.putText(frame, word, (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow('capture', frame)
                translated = True
                pen_location = (0, 0)

                # if in testmode write to file
                if self.test_mode:
                    with open(self.output_file, "a", encoding="utf-8") as f:
                        f.write(f"{choosen_word}\n")

                else:
                    print(f"choosen word: {choosen_word}")
                    translation = translate(choosen_word)
                    print(f"translation: {translation}")
                    word_to_speak(translation)
            if self.test_mode:
                # delay in ms for 30 fps to match video
                delay = int(1000 / 30)
            # Exit if 'q' is pressed
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
