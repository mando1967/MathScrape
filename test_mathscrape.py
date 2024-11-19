import os
import sys
from PIL import Image
import numpy as np

# Set environment variable to avoid OpenMP initialization error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathscrape import MathScrape

def test_mathscrape():
    """Test the MathScrape class with a test image"""
    print("\nStarting MathScrape test...\n")
    
    # Initialize MathScrape
    try:
        scraper = MathScrape()
        
        # Get the path to the test image
        test_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(test_dir, 'test_math_problems.png')
        
        # Process image
        results = scraper.extract_math_expressions(image_path)
        
        # Basic validation
        if results is None:
            raise ValueError("Results should not be None")
        if len(results) == 0:
            raise ValueError("Should detect at least one math expression")
        
        # Validate result structure
        for result in results:
            if 'problem_id' not in result:
                raise ValueError("Each result should have a problem_id")
            if 'text' not in result:
                raise ValueError("Each result should have extracted text")
            if 'bbox' not in result:
                raise ValueError("Each result should have a bounding box")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please ensure:")
        print("1. Tesseract OCR is properly installed")
        print("2. The test image exists and is readable")
        print("3. All required Python packages are installed")
        raise

if __name__ == "__main__":
    test_mathscrape()
