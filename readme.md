[![Watch the demo](https://img.youtube.com/vi/J5BZkDXeA2k/hqdefault.jpg)](https://www.youtube.com/watch?v=J5BZkDXeA2k)
# OCR Translation

A real-time OCR (Optical Character Recognition) tool using OpenCV and Tesseract.  
It recognizes a blue-tipped pen held under a word in a video feed, extracts the word using OCR, translates it from French to English using the Google Translate API, and reads it aloud.

---

## Features

- Detects a blue-tipped pen in a live video or video file
- Extracts French words using OCR when the pen stays under them
- Translates words to English via Google Translate
- Reads translations aloud using text-to-speech
- Supports test mode to process video files and save results to a file

---

## Demo Instructions

- Run ocr-translate in terminal to start the app
- Press `q` to quit the application
- Use a pen or object with a distinct blue tip
- Hold it under a clearly printed French word for ~1 second to trigger translation

---

## Installation

Run: pip install -e
Or for dev mode: pip install -e .[dev]

### Prerequisites

- Python 3.7+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and added to your system PATH
- **French language support for Tesseract**

#### Install Tesseract with French support:

- **Linux (Ubuntu/Debian):**

  sudo apt install tesseract-ocr tesseract-ocr-fra

- **macOS (using Homebrew):**

  brew install tesseract
  brew install tesseract-lang # or manually install fra.traineddata if needed

- **Windows:**
  1. Install Tesseract from [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
  2. Download `fra.traineddata` from [tessdata](https://github.com/tesseract-ocr/tessdata)
  3. Place it in your Tesseract `tessdata` directory (e.g., `C:\Program Files\Tesseract-OCR\tessdata`)

### Environment variables

Create a `.env` file in your project directory and add your Google Translate API key:

```dotenv
GOOGLE_TRANSLATE_API_KEY=your_api_key_here
```

---

## Usage

### Real-time mode (default webcam)

To run the app using your webcam write in terminal:

ocr-translate in terminal to start the app

This starts the live feed. When you hold a blue pen under a word and keep it still for 1 sec, the app will:

- Detect the pen
- Read the word using OCR
- Translate it to English
- Say it aloud

### Test mode (video file with output logging)

To run the app on a video file and save translations to a text file:

Run the script:

ocr-translate video_file.mp4

Words under the pen will be processed and saved to `output.txt`.

## Dependencies

See `requirements.txt`:
