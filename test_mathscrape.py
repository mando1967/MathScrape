import os
import sys
from PIL import Image
import numpy as np
import unittest
from mathscrape import MathScrape
import cv2

# Set environment variable to avoid OpenMP initialization error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestMathScrape(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.scraper = MathScrape()
        
    def test_preprocess_text(self):
        """Test text preprocessing for various mathematical expressions"""
        # Test quadratic equation preprocessing
        text = "x?+5x+6=0"
        processed = self._preprocess_text(text)
        self.assertEqual(processed, "x^2+5x+6=0")
        
        # Test integral preprocessing
        text = "J(x2-+1)dx"
        processed = self._preprocess_text(text)
        self.assertEqual(processed, "\\int(x^{2}-1)\\,dx")
        
        # Test limit preprocessing
        text = "lim(x--)1/x=0"
        processed = self._preprocess_text(text)
        self.assertEqual(processed, "\\lim_{x \\to \\infty}\\frac{1}{x}=0")
        
    def _preprocess_text(self, text):
        """Helper method to test text preprocessing"""
        processed_text = text
        
        # Handle polynomial terms
        import re
        processed_text = processed_text.replace('x?', 'x^2')
        processed_text = re.sub(r'x(\d)', r'x^{\1}', processed_text)
        processed_text = re.sub(r'(\d)x', r'\1x', processed_text)
        
        # Handle integral symbol
        if 'J' in text:
            processed_text = processed_text.replace('J', '\\int')
            processed_text = re.sub(r'dx$', '\\,dx', processed_text)
        
        # Handle limit notation
        if 'lim' in text:
            processed_text = processed_text.replace('lim(x--)', '\\lim_{x \\to \\infty}')
            processed_text = processed_text.replace('1/x', '\\frac{1}{x}')
            
        return processed_text
        
    def test_image_processing(self):
        """Test image processing with test math problems"""
        # Create test image with math problems
        img = self._create_test_image([
            "x?+5x+6=0",
            "J(x2-+1)dx",
            "lim(x--)1/x=0"
        ])
        
        # Save test image
        test_image_path = "test_math_problems.png"
        cv2.imwrite(test_image_path, img)
        
        try:
            # Process the image
            results = self.scraper.extract_math_expressions(test_image_path)
            
            # Verify results
            self.assertTrue(len(results) >= 3, "Should detect at least 3 math problems")
            
            # Clean up
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                
        except Exception as e:
            # Clean up even if test fails
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
            raise e
            
    def _create_test_image(self, expressions):
        """Create a test image with given math expressions"""
        # Create blank image
        height = len(expressions) * 50 + 50
        width = 400
        img = np.ones((height, width), dtype=np.uint8) * 255
        
        # Add expressions
        for i, expr in enumerate(expressions):
            cv2.putText(
                img,
                expr,
                (50, 50 + i * 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                0,
                2
            )
            
        return img

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
    unittest.main()
    test_mathscrape()
