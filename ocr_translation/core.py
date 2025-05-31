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

import time
import logging
import cv2
import pytesseract
import numpy as np
from dotenv import load_dotenv
from .utils import dist, translate, word_to_speak, empty, mask_page_outside
from .image_tools import preprocess_image
from ultralytics import YOLO
import winsound
# pylint: disable = no-member
load_dotenv()  # load the google translate key as env variable

# logging configuration
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]%(message)s")


class CameraOCR:
    """
    A class that captures video from the webcam, allows live preview, recognize a pen
    and performs OCR (Optical Character Recognition) when the pen stays long enough under a word.
    It then translate the word and reads it out loud.
    """

    def __init__(self, source=1, output_file=None):  # default camera is 1
        """
        Initializes the CameraOCR by creating a video capture object for the default webcam.
        """
        self.test_mode = output_file is not None  # return true if test mode is on
        self.cap = cv2.VideoCapture(source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logging.info("Actual resolution: %s x %s", actual_width, actual_height)

        if not self.cap.isOpened():
            if self.test_mode:
                raise IOError("Cannot open video file.")
            else:
                raise IOError("Cannot open camera.")
        self.output_file = output_file

        # improt page detection model
        self.page_detector = YOLO("models/page_detecter_v2obb.pt")
        # Configurable constants:

        # How many pixels considered a move of the pen
        self.distance_threshold = 30
        # How long (in secs) the pen needs to stay for a translation
        self.time_still_treshold = 0.2
        # Delay in ms for 30 fps to match video
        self.test_mode_delay = int(1000 / 30)
        # Percent of certainty needed to detect a word
        self.word_certainty = 60
        # certainty threshold for page detection between 0-1
        self.word_certainty = 0.6
        # min area for pen detection to prevent false positive due to noise
        self.min_area = 10
        # skip loop rate to improve performance
        self.frame_skip_rate = 10  # Process every 10rd frame

    def run(self):
        """
        Starts the live camera feed.
        - Press 'q' to quit the application and close the window.
        """
        if self.test_mode:
            logging.info("Processing video in test mode")

        time_since_last_move = time.time()
        time_still = 0
        last_location = None
        pen_location = None  # topmost point in the pen
        translated = True  # signify that the current word was translated already
        delay = 1
        frame_count = 0
        results = None
        count = 0  # tmp
        while True:
            logging.debug("pen location: %s", pen_location)

            ret, frame = self.cap.read()
            # cv2.imwrite("camera_snapshot.jpg", frame)
            if not ret:
                if self.test_mode:
                    logging.info("Video playback end.")
                else:
                    logging.warning("Video ended or failed to grab frame.")
                break

            # page detection bounding boxes
            if frame_count % self.frame_skip_rate == 0:
                count += 1
                results = self.page_detector(frame, show=False, verbose=False)

            choosen_page = None
            pen_location = self.detect_pen_location(frame)
            pen_already_detected_inside_page = False

            if results[0].obb is not None:
                for obb, conf in zip(results[0].obb.xyxyxyxy, results[0].obb.conf):
                    conf_value = float(conf)  # Confidence value
                    if conf_value < self.word_certainty:
                        continue
                    obb_np = obb.cpu().numpy()  # Convert PyTorch tensor to numpy array

                    if pen_location is not None and not pen_already_detected_inside_page:

                        # check if the pen is inside a page
                        pen_location = (
                            int(pen_location[0]), int(pen_location[1]))
                        inside = cv2.pointPolygonTest(
                            obb_np, pen_location, False)
                        if inside >= 0:  # if the pen is inside or on the edge of a page
                            choosen_page = obb_np
                            pen_already_detected_inside_page = True

                    # Draw the detected OBBs on the frame

                    pts = obb_np.reshape((-1, 1, 2)).astype(int)
                    cv2.polylines(frame, [pts], isClosed=True,
                                  color=(0, 255, 0), thickness=2)

            if not pen_already_detected_inside_page:
                pen_location = None

            # how long the pen was still
            time_still = (
                time.time() - time_since_last_move
            )
            # if pen_location:
            # print(f"time still {time_still}")
            # if pen recently moved or is out of the frame
            # if pen_location:
            # print(dist(pen_location, last_location))
            if (
                pen_location is None
                or dist(pen_location, last_location) > self.distance_threshold
            ):
                time_since_last_move = time.time()
                last_location = pen_location
                translated = False
            # if the pen stays long enough under a word
            elif (
                pen_location is not None
                and not translated
                and time_still > self.time_still_treshold
            ):
                winsound.Beep(1000, 200)
                start_pipeline = time.time()
                # mask everything that's outside the choosen page
                masked = mask_page_outside(frame, choosen_page)
                cv2.namedWindow("masked", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("masked", 960, 540)
                cv2.imshow("masked", masked)
                choosen_word = self.get_word_to_translate(masked, pen_location)
                translated = True
                # print(f"pipe line duration {time.time()-start_pipeline}")
                # if in testmode write the word to file
                if self.test_mode:
                    with open(self.output_file, "a", encoding="utf-8") as f:
                        f.write(f"{choosen_word}\n")
                # else, print to terminal
                else:
                    print(f"choosen word: {choosen_word}")
                    en_translation = translate(choosen_word, "en")
                    print(f"English: {en_translation}")
                    word_to_speak(en_translation)
                    he_translation = translate(choosen_word, "he")
                    print(f"Hebrew: {he_translation[::-1]}")
                    with open("Words learned.txt", "a", encoding="utf-8") as f:
                        f.write(
                            f"{choosen_word} - {en_translation} - {he_translation}\n")
            # show the video stream
            cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Live Feed", 960, 540)
            cv2.imshow("Live Feed", frame)
            if self.test_mode:
                delay = self.test_mode_delay
            # Exit if 'q' is pressed
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                logging.info("User exits program.")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Camera released and windows closed.")
        print(count)

    def detect_pen_location(self, frame):
        """
        Detects the pen tip location in the video frame based on a blue color filter.

        This method converts the input frame to HSV color space and applies a mask to isolate
        blue regions. It then finds contours in the mask and selects the largest one. If the
        contour area exceeds a minimum threshold (`self.min_area`), it returns the highest
        point of that contour, which is assumed to be the tip of a pen.

        Args:
            frame (np.ndarray): The current video frame (BGR format) captured from the camera.

        Returns:
            tuple or None: (x, y) coordinates of the detected pen tip if found, otherwise None.
        """
        detect_pen_start = time.time()
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
            # return the highest point of the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= self.min_area:
                # print(
                #     f"detect pen location took {time.time()-detect_pen_start}")
                return tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        return None

    def get_word_to_translate(self, frame, pen_location: tuple) -> str:
        """
        Detects the closest French word below a given pen location in the video frame using OCR.

        This method processes the input frame with Tesseract OCR to detect all visible words.
        It then finds the word with the highest confidence score that is below the pen tip
        and closest in distance to it.

        Args:
            frame (np.ndarray): The current video frame (BGR format) captured from the camera.
            pen_location (tuple): A (x, y) tuple representing the position of the pen tip.

        Returns:
            str: The closest word found under the pen location, or an empty string if none match.

        Raises:
            pytesseract.TesseractNotFoundError: If Tesseract OCR is not installed or not found.
            Exception: If any other OCR-related error occurs.
        """
        # apply preprocessing
        processed_image = preprocess_image(frame, pen_location)
        # find the new pen location after preprocessing
        pen_location = self.detect_pen_location(processed_image)
        choosen_word = ""
        min_dist = float("inf")
        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        try:
            tesseract_start = time.time()
            data = pytesseract.image_to_data(
                image_rgb, lang="fra", output_type=pytesseract.Output.DICT
            )
            print(f"tesseract took {time.time()-tesseract_start}")
        except pytesseract.TesseractNotFoundError:
            logging.error(
                "Tesseract is not installed or not found in system path.")
            raise
        except Exception as e:
            logging.error("OCR failed: %s", e)
            raise

        n = len(data["text"])
        for i in range(n):
            if int(data["conf"][i]) > self.word_certainty and not empty(
                data["text"][i]
            ):
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                word = data["text"][i]
                word_loc = (int(x + w / 2), y + h)  # middle buttom of the word
                if y < pen_location[1] and dist(pen_location, word_loc) < min_dist:
                    min_dist = dist(pen_location, word_loc)
                    choosen_word = word
                # temporary:
                cv2.circle(processed_image, word_loc, 3, (0, 255, 0), -1)
                cv2.circle(processed_image, pen_location, 1, (0, 0, 255), -1)
                cv2.rectangle(processed_image, (x, y),
                              (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    processed_image, word, (x,
                                            y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )
            # cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("capture", 960, 540)
            # show the image that was sent to tesseract
            cv2.imshow("capture", processed_image)
        return choosen_word
