"""Tests for the utility functions in utils.py"""
import unittest
from ocr_translation import utils


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""

    def test_dist_between_two_points(self):
        """Test the dist func."""
        self.assertEqual(utils.dist((0, 0), (3, 4)), 5.0)

    def test_dist_with_none(self):
        """Test the dist func."""
        self.assertEqual(utils.dist(None, (1, 2)), float("inf"))
        self.assertEqual(utils.dist((1, 2), None), float("inf"))

    def test_empty(self):
        """Test the empty func."""
        self.assertTrue(utils.empty(""))
        self.assertTrue(utils.empty("   "))
        self.assertFalse(utils.empty("bonjour"))

    def test_translate(self):
        """Test the translate func."""
        self.assertEqual(utils.translate("chat"), "cat")


if __name__ == "__main__":
    unittest.main()
