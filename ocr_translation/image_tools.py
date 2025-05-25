"""
This module provides utility functions for preprocessing images in preparation
for OCR (Optical Character Recognition). Functions include deskewing...
 These are designed to align... input to improve OCR accuracy and consistency.

Functions:
- get_skew_angle(image, debug=False): Estimates the skew angle of text in the image.

"""
import logging
import numpy as np
import cv2


# pylint: disable = no-member

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]%(message)s")


def preprocess_image(image: np.ndarray, pen_location: tuple) -> np.ndarray:
    """
    run all preprocessing steps before the image is sent to tesseract
    """
    logging.info("starting preprocessing...")
    # step 1: Auto zoom in for optimal OCR result

    # Calculate the amount of zoom needed
    text_height = estimate_letter_height(image, pen_location)

    target_height = 30
    min_zoom = 1.0
    max_zoom = 4.0

    if text_height > 0:
        zoom = target_height / text_height
        zoom = min(max(zoom, min_zoom), max_zoom)
    else:
        zoom = 2.0  # fallback

    logging.info("Text height recognized: %s", text_height)
    logging.info("performed %sX zoom", zoom)

    # Apply the zoom by croping and resizing
    x, y = pen_location
    h, w = image.shape[:2]

    crop_width = int(w / zoom)
    crop_height = int(h / zoom)
    x1 = max(x - crop_width // 2, 0)
    y1 = max(y - crop_height // 2, 0)
    x2 = min(x + crop_width // 2, w)
    y2 = min(y + crop_height // 2, h)
    cropped = image[y1:y2, x1:x2]

    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

    # step 2: deskew image

    angle = get_skew_angle(zoomed)
    if abs(angle) > 0.5:
        center = (zoomed.shape[1] // 2, zoomed.shape[0] // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewd = cv2.warpAffine(
            zoomed, m, (zoomed.shape[1], zoomed.shape[0]),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    else:
        deskewd = zoomed

    return deskewd


def get_skew_angle(image: np.ndarray, debug: bool = False) -> float:
    """
    Calculates the skew angle of text in an image using contour-based analysis of the lines.

    This function applies preprocessing (grayscale, Gaussian blur, thresholding, and dilation)
    to detect lines of text. It finds contours, estimates their orientation using
    minimum-area rectangles, filters noise and outlier angles, and returns the average
    skew angle. If `debug` is True, it also displays the intermediate processing steps.

    Args:
        image (np.ndarray): The input BGR image to analyze.
        debug (bool, optional): If True, displays debug visualizations. Defaults to False.

    Returns:
        float: Estimated skew angle in degrees. A positive angle means counter-clockwise rotation
               is needed to deskew the image.
    """
    logging.debug("Starting skew angle detection...")
    # convert to grey
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur image to reduce noise
    blur = cv2.GaussianBlur(grey, (9, 9), 0)
    # apply a treshhold
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    logging.debug("Applied thresholding and blurring.")
    # dilatation with a kernel largen on the x axis in order to merge all words in a line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
    dilate = cv2.dilate(thresh, kernel, iterations=3)
    # find lines contours
    contours, _ = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    logging.debug("Found %s contours before filtering.", len(contours))
    # filter smaller contour that comes from noise
    contours = [c for c in contours if cv2.contourArea(c) > 500]
    logging.debug("%s contours remaining after area filter.", len(contours))
    if debug:
        # draw contours
        draw_result = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)

    # list all lines angles
    line_angles = []
    for contour in contours:
        min_area_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(min_area_rect)
        box = box.astype(int)
        line_angles.append(min_area_rect[-1])
        if debug:
            cv2.drawContours(draw_result, [contour], -1, (0, 255, 0), 5)
            cv2.drawContours(draw_result, [box], 0, (0, 0, 255), 2)

    line_angles = [a for a in line_angles if a not in [0.0, 45.0, 90.0]]
    line_angles = remove_outliers_iqr(line_angles)
    logging.debug("lines angles: %s", line_angles)
    logging.debug("After filtering %s lines detected", len(line_angles))
    if not line_angles:
        logging.warning("No valid angles found — returning 0.")
        return 0.0
    # finding the image angle
    image_angle = sum(line_angles) / len(line_angles)
    # assuming the angle isnt larger than 45 we decide the skew direction
    if image_angle > 45:
        image_angle = image_angle-90

    if debug:
        # rotate picture
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, image_angle, 1.0)
        rotated = cv2.warpAffine(
            image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # show processing result
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)  # Make window resizable
        cv2.resizeWindow('Image', 600, 800)
        cv2.imshow('Image', draw_result)
        # show image after rotation
        cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)  # Make window resizable
        cv2.resizeWindow('rotated', 600, 800)
        cv2.imshow('rotated', rotated)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    logging.info("Detected skew angle: %s°", image_angle)
    return image_angle


def estimate_letter_height(image, pen_location, search_radius=200, debug=False):
    """
    Estimate average height of black contours (letters) near the pen location.
    Optionally displays debug visualizations.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_pen, y_pen = pen_location
    heights = []

    debug_image = image.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        # find all letters within the search zone
        if abs(cx - x_pen) < search_radius and abs(cy - y_pen) < search_radius:
            if 1 < h < 100:
                heights.append(h)
                if debug:
                    cv2.rectangle(debug_image, (x, y),
                                  (x + w, y + h), (0, 255, 0), 2)

    if debug:
        # Draw pen location and search radius
        cv2.circle(debug_image, (x_pen, y_pen), 5, (0, 0, 255), -1)
        cv2.circle(debug_image, (x_pen, y_pen), search_radius, (255, 0, 0), 2)

        cv2.namedWindow("letters search radius", cv2.WINDOW_NORMAL)
        cv2.imshow("letters search radius", debug_image)

        cv2.namedWindow("Threshold (black letters)", cv2.WINDOW_NORMAL)
        cv2.imshow("Threshold (black letters)", thresh)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return sum(heights) / len(heights) if heights else 0


def remove_outliers_iqr(values, iqr_factor=1.5):
    """
    Removes outliers from a list of numbers using the IQR method.

    Args:
        values (list or np.ndarray): List of numeric values.
        iqr_factor (float): Controls how aggressive the filtering is (default 1.5).

    Returns:
        list: Values with outliers removed.
    """
    if not values:
        return []

    q1 = np.percentile(values, 25)  # The limit between Q1 to Q2
    q3 = np.percentile(values, 75)  # The limit between Q3 to Q4
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr

    return [x for x in values if lower_bound <= x <= upper_bound]
