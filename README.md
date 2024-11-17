# MathScrape

MathScrape is a Python application that extracts mathematical expressions from images and converts them to both text and LaTeX format. It uses advanced computer vision and OCR techniques to identify and process mathematical content.

## Features

- Image preprocessing for optimal text extraction
- Automatic detection of mathematical regions
- Conversion of mathematical expressions to LaTeX
- Text extraction using Tesseract OCR
- Visual results with highlighted math regions
- Support for multiple math expressions in a single image

## Prerequisites

1. Python 3.7 or higher
2. Tesseract OCR installed on your system
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Make sure to add Tesseract to your system PATH

## Installation

1. Create a new conda environment:
   ```bash
   conda create -n mathscrape python=3.10
   conda activate mathscrape
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Activate the conda environment:
   ```bash
   conda activate mathscrape
   ```

2. Run the application:
   ```bash
   python mathscrape.py
   ```

3. When prompted, enter the path to your image containing mathematical expressions.

4. The application will:
   - Display the detected math regions
   - Show the extracted text
   - Provide LaTeX representations of the mathematical expressions
   - Visualize the results with highlighted regions

## Example

Input: An image containing mathematical equations
Output:
```
Extracted Mathematical Expressions:
----------------------------------------

Region 1:
Text: x^2 + 5x + 6 = 0
LaTeX: x^{2} + 5x + 6 = 0

Region 2:
Text: ∫(x^2 + 1)dx
LaTeX: \int(x^{2} + 1)dx
```

## Notes

- For best results, ensure that:
  - The image is clear and well-lit
  - Mathematical expressions are properly aligned
  - There is good contrast between text and background
  - The image is in a common format (PNG, JPG, etc.)

## Troubleshooting

1. If Tesseract is not found:
   - Ensure Tesseract is properly installed
   - Verify the path in `mathscrape.py` matches your Tesseract installation path

2. If math regions are not detected:
   - Try adjusting the image quality
   - Ensure sufficient contrast in the image
   - Check if the mathematical expressions are clear and readable

## License

This project is licensed under the MIT License - see the LICENSE file for details.
