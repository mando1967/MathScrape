# MathScrape

A Python application for detecting and extracting mathematical expressions from images, with support for multiple problems and LaTeX conversion.

## Features

- Detects multiple mathematical problems in a single image
- Extracts text and converts to LaTeX
- Adaptive region detection with configurable tolerances
- Beautiful visualization of detected regions
- Support for complex mathematical notation including limits, integrals, and equations

## Example Output

![Math Problem Detection](visualization.png)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mando1967/MathScrape.git
cd MathScrape
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate mathscrape
```

3. Install Tesseract OCR:
- Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Make sure it's installed at: `C:\Program Files\Tesseract-OCR\tesseract.exe`

## Usage

Run the test script to process a sample image:
```python
python test_mathscrape.py
```

Or use in your own code:
```python
from mathscrape import MathScrape

# Initialize
scraper = MathScrape()

# Process an image
results = scraper.extract_math_expressions('path/to/your/image.png')

# Results contain text and LaTeX for each detected problem
for result in results:
    print(f"Problem {result['problem_id']}:")
    print(f"Text: {result['text']}")
    print(f"LaTeX: {result['latex']}\n")
```

## Configuration

The MathScrape class includes configurable parameters:
- `horizontal_tolerance`: Controls merging of horizontally adjacent regions (default: 0.2)
- `vertical_tolerance`: Controls separation of problems (default: 0.8)

## Requirements

- Python 3.10+
- Tesseract OCR
- See `environment.yml` for complete list of dependencies

## License

MIT License - feel free to use and modify as needed!
