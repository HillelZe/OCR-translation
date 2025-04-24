"""Tests for the core functions in core.py"""
import unittest
from ocr_translation import CameraOCR


class TestCameraOCR(unittest.TestCase):
    """Tests for the CameraOCR class"""

    def test_init_with_invalid_source(self):
        """Test invalid video test file"""
        with self.assertRaises(OSError):
            CameraOCR(source="invalid.mp4")  # This file shouldn't exist


if __name__ == "__main__":
    unittest.main()
