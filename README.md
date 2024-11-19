# MathScrape

A Python application for extracting and converting mathematical expressions from images to LaTeX format.

## Features

- Extracts mathematical expressions from images using OCR
- Converts recognized text to LaTeX format
- Supports various mathematical notations:
  - Quadratic equations (e.g., "x^2+5x+6=0")
  - Integrals (e.g., "∫(x^2-1)dx")
  - Limits (e.g., "lim_{x→∞} 1/x=0")
- Handles common OCR misrecognitions and formatting issues
- Visualizes detected math regions in the image

## Requirements

- Python 3.7+
- OpenCV
- Tesseract OCR
- pix2tex
- Other dependencies listed in `requirements.txt`

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
- Windows: Run `install_tesseract.py`
- Other platforms: Follow [Tesseract installation guide](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

1. Run the application:
```bash
python mathscrape.py
```

2. Enter the path to your image when prompted

3. The application will:
   - Detect mathematical expressions in the image
   - Convert them to LaTeX format
   - Display the results
   - Generate a visualization of detected regions

## Recent Improvements

- Enhanced limit notation handling (e.g., "lim(x→∞)")
- Improved integral symbol recognition (handles both "∫" and "J")
- Better polynomial term processing (handles squared terms and coefficients)
- Added comprehensive test suite for mathematical expression preprocessing
- Fixed common OCR misrecognition issues

## Testing

Run the test suite:
```bash
python test_mathscrape.py
```

## Contributing

Feel free to submit issues and enhancement requests!
