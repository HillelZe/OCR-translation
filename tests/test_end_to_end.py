"""
End-to-end test for the OCR translation app using a sample video.

This test runs the main OCR script on a sample video file, reads the generated
output, compares the results to expected French words, and verifies correctness.
"""

import subprocess
import sys
import os
import logging


def test_end_to_end_success():
    """
    Tests that the expected French words are correctly recognized
    from the provided video sample and written to the output file.
    """
    # Run the main script with the video input
    try:
        subprocess.run([sys.executable, "main.py",
                       "test_video.mp4"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error("Failed to run main.py on video: %s", e)
        assert False, "OCR translation app crashed during test"

    # Read the results from output.txt
    try:
        with open("output.txt", encoding="utf-8") as f:
            results = f.read().splitlines()
    except IOError as e:
        logging.error("Could not read output.txt: %s", e)
        assert False, "Missing or unreadable output file"

    expected_results = ["Profitez", "français", "faites", "possible"]
    for word in expected_results:
        assert word in results, f"Expected '{word}' to be in output."

    # Cleanup output file
    try:
        os.remove("output.txt")
    except PermissionError as e:
        logging.warning("Could not delete output.txt: %s", e)
