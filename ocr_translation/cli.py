"""
Command-line interface for the OCR Translation app.

This file allows users to run the app from the terminal using a video file
or fallback to the webcam if no file is provided.
"""

import argparse
import os
from .core import CameraOCR


def main():
    """
    Entry point for the ocr-translate CLI command.
    Parses arguments and launches the OCR app.
    """
    parser = argparse.ArgumentParser(description="OCR Translator")
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to a video file for test mode. If omitted, webcam will be used.",
    )
    args = parser.parse_args()

    if args.video:
        if not os.path.isfile(args.video):
            print(f"File not found: {args.video}")
            return
        ocr = CameraOCR(source=args.video, output_file="output.txt")
    else:
        ocr = CameraOCR()  # fallback to webcam

    ocr.run()
