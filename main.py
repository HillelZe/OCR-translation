"""
Entry point for the OCR translation app.

Parses an optional video file argument and initializes the OCR process using
CameraOCR. If no video path is provided, the webcam is used as the input source.
In test mode (video input), results are written to 'output.txt'.
"""

import argparse
import os
from ocr_translation.core import CameraOCR


def main():
    """
    Parses CLI arguments and runs the OCR translation app.
    """
    parser = argparse.ArgumentParser(description="OCR Translator")
    parser.add_argument("video", nargs="?", help="Path to video file")
    args = parser.parse_args()

    if args.video:
        if not os.path.isfile(args.video):
            print(f"File not found: {args.video}")
            return
        ocr = CameraOCR(source=args.video, output_file="output.txt")
    else:
        ocr = CameraOCR()  # fallback to live camera if no video file is given

    ocr.run()


if __name__ == "__main__":
    main()
