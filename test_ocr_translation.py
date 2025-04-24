"""
Runs an end-to-end test for the OCR translation app using a sample video.

This script performs the following steps:
1. Runs the main OCR application (`main.py`) with a test video file (`test_video.mp4`).
2. As the main program run in test mode the recognized word will be written to an output file.
3. Reads the output file and compares the results an expected list of results and counts matches.
4. Prints the number of successful matches to the console.
5. Deletes the `output.txt` file after the test to clean up.

Raises:
    subprocess.CalledProcessError: If running the OCR application fails.
    IOError: If reading the output file fails.
    PermissionError: If the script is denied permission to delete the output file.
"""
import sys
import os
import logging
import subprocess


def main():
    """
    Executes the end-to-end OCR test pipeline using a sample video.

    This function runs the main OCR app, validates the output against expected results,
    prints the success rate, and cleans up the output file afterward.
    """

    # Run the main script with a video file
    subprocess.run([sys.executable, "main.py", "test_video.mp4"], check=True)

    # check the output
    expected_results = ["Profitez", "français",
                        "faites",  "possible",]
    results = []
    matches = 0
    try:
        with open("output.txt", encoding="utf-8") as f:
            results = f.read().splitlines()
    except IOError as e:
        logging.error("couldn't write to output file. %s", e)

    # print results
    for word in expected_results:
        if word in results:
            matches += 1
    print(f"{matches}/{len(expected_results)} succeeded!")

    # Delete the output file after the test
    try:
        if os.path.exists("output.txt"):
            os.remove("output.txt")
            logging.info("Output file deleted successfully")
        else:
            logging.warning("Output file not found")
    except PermissionError as e:
        logging.error("Permission denied when deleting output file. %s", e)


if __name__ == "__main__":
    main()
