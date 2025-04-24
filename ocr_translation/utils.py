"""
Utility functions for the OCR translation app.

This module includes general-purpose helper functions that support the core OCR logic:
- Distance calculation between points
- String checks
- Translation using the Google Translate API
- Text-to-speech output using pyttsx3

These functions are imported and used in the camera module and are intended
to remain stateless and reusable.

Dependencies:
- requests
- pyttsx3
- python-dotenv
"""


import math
import os
import logging
import requests
import pyttsx3


def dist(p1: tuple, p2: tuple) -> float:
    """
    Calculate the Euclidean distance between two 2D points.

    Args:
        p1: A tuple (x, y) representing the first point.
        p2: A tuple (x, y) representing the second point.

    Returns:
        The distance as a float.
    """
    if p1 is None or p2 is None:
        return float("inf")
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def translate(word: str) -> str:
    """
    Translate a French word into English using Google Translate API.

    Args:
        word: A string containing a single French word to translate.

    Returns:
        The translated English word as a string.

    Raises:
        ValueError: If the API key is not set.
        requests.RequestException: If the HTTP request fails.
    """
    api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing environment variable: GOOGLE_TRANSLATE_API_KEY")
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {"key": api_key, "source": "fr", "target": "en", "q": word}
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()  # Raise error for bad HTTP codes
        data = response.json()
        translated_text = data["data"]["translations"][0]["translatedText"]
        return translated_text
    except requests.RequestException as e:
        logging.error("Translation API error: %s", e)
        raise


def word_to_speak(word: str) -> None:
    """
    Gets a word as a string and reads it out loud.
    """
    try:
        engine = pyttsx3.init()
        # get an english voice
        for voice in engine.getProperty("voices"):
            if "en" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        engine.say(word)
        engine.runAndWait()
    except Exception as e:
        logging.error("Text to sound failed. %s", e)


def empty(s: str) -> bool:
    """
    Checks if a string is empty or consists only of whitespace.

    Args:
        s (str): The input string.

    Returns:
        bool: True if the string is empty or only whitespace, False otherwise.
    """
    return s.strip() == ""
