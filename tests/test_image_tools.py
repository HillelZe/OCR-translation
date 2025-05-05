import cv2
import os
from ocr_translation.image_tools import get_skew_angle
# pylint: disable = no-member


def test_get_skew_angle_on_skew_test1():
    img_path = os.path.join(os.path.dirname(__file__),
                            "images", "skew_test1.jpeg")
    image = cv2.imread(img_path)
    angle = get_skew_angle(image)
    expected_angle = 7.76
    assert abs(angle - expected_angle) < 1.0


def test_get_skew_angle_on_skew_test2():
    img_path = os.path.join(os.path.dirname(__file__),
                            "images", "skew_test2.jpeg")
    image = cv2.imread(img_path)
    angle = get_skew_angle(image)
    expected_angle = -18.43
    assert abs(angle - expected_angle) < 1.0
