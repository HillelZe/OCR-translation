import argparse
import os
from ocr_translation import CameraOCR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Translator")
    parser.add_argument("video", nargs="?", help="Path to video file")
    args = parser.parse_args()

if args.video:
    if not os.path.isfile(args.video):
        print(f"File not found: {args.video}")
    else:
        ocr = CameraOCR(source=args.video, output_file="output.txt")
        ocr.run()
else:
    ocr = CameraOCR()  # fallback to live camera if no video file is given
    ocr.run()
