import logging
import cv2
import numpy as np
# pylint: disable = no-member

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]%(message)s")


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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=3)
    # find lines contours
    contours = cv2.findContours(
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

    # filter mistaken contours(maybe should filter outliers instead?)
    line_angles = [a for a in line_angles if a not in [0.0, 45.0, 90.0]]

    logging.debug("After filtering %s lines detected", len(line_angles))
    if not line_angles:
        logging.warning("No valid angles found — returning 0.")
        return 0.0
    # finding the image angle
    image_angle = sum(line_angles) / len(line_angles)
    # assuming the angle isnt larger than 45 we decide the skew direction
    if image_angle > 45:
        image_angle = (90 - image_angle)*-1

    if debug:
        # rotate picture
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, image_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # show processing result
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)  # Make window resizable
        cv2.resizeWindow('Image', 600, 800)
        cv2.imshow('Image', draw_result)
        # show image after rotation
        cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)  # Make window resizable
        cv2.resizeWindow('rotated', 600, 800)
        cv2.imshow('rotated', rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    logging.info("Detected skew angle: %s°", image_angle)
    return image_angle
