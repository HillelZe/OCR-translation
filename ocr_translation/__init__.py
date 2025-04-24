"""
OCR Translation package initializer.

This file exposes key classes and functions for easy import,
including the main `CameraOCR` class and translation/speech utilities.

Example:
    from ocr_translation import CameraOCR, translate, word_to_speak
"""
# ocr_translation/__init__.py
from .core import CameraOCR
from .utils import translate, word_to_speak
