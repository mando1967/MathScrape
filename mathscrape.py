import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
from pix2tex.cli import LatexOCR
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

class MathScrape:
    def __init__(self):
        """Initialize MathScrape with OCR and LaTeX recognition capabilities"""
        # Check if Tesseract is installed
        if not os.path.exists('C:\\Program Files\\Tesseract-OCR\\tesseract.exe'):
            raise RuntimeError(
                "Tesseract not found. Please ensure it is installed at 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'"
            )
            
        # Initialize LaTeX OCR model
        print("Initializing LaTeX OCR model...")
        try:
            from pix2tex.cli import LatexOCR
            import torch
            import warnings
            warnings.filterwarnings('ignore')  # Suppress warnings
            
            # Disable gradients for inference
            torch.set_grad_enabled(False)
            
            # Create a test image with a simple math expression
            test_image = Image.new('RGB', (100, 50), color='white')
            test_draw = ImageDraw.Draw(test_image)
            test_draw.text((10, 10), "1+1=2", fill='black')
            
            # Initialize model
            self.latex_ocr = LatexOCR()
            
            # Test the model
            test_result = self.latex_ocr(test_image)
            print("LaTeX OCR model initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize LaTeX OCR model: {str(e)}")
            self.latex_ocr = None
        
        # Ensure Tesseract path is set correctly
        if os.name == 'nt':  # Windows
            if not os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
                raise RuntimeError(
                    "Tesseract not found. Please ensure it is installed at 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'"
                )
        
    def preprocess_image(self, image_path):
        """
        Preprocess the input image for better math region detection
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray: Preprocessed binary image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def preprocess_for_latex(self, image):
        """
        Special preprocessing for LaTeX recognition
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Preprocessed image optimized for LaTeX recognition
        """
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if not already
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Ensure array is uint8
        img_array = img_array.astype(np.uint8)
            
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_array)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Binarize using Otsu's method
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if background is dark
        if np.mean(binary[0:10, 0:10]) < 127:  # Check top-left corner
            binary = cv2.bitwise_not(binary)
            
        # Convert back to PIL Image in RGB mode (pix2tex expects RGB)
        rgb_array = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_array)
        
        # Resize if too small
        if pil_image.size[0] < 32 or pil_image.size[1] < 32:
            scale = max(32/pil_image.size[0], 32/pil_image.size[1])
            new_size = tuple(int(dim * scale) for dim in pil_image.size)
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
            
        # Add padding
        padded = Image.new('RGB', (pil_image.size[0] + 20, pil_image.size[1] + 20), color='white')
        padded.paste(pil_image, (10, 10))
            
        return padded

    def detect_math_regions(self, preprocessed_image):
        """
        Detect regions containing mathematical expressions
        
        Args:
            preprocessed_image (numpy.ndarray): Preprocessed image
            
        Returns:
            list: List of bounding boxes for detected math regions
        """
        # Find contours
        contours, _ = cv2.findContours(
            preprocessed_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Get image dimensions
        height, width = preprocessed_image.shape
        min_area = (width * height) * 0.001  # Adaptive minimum area
        
        # Filter contours based on area and aspect ratio
        math_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h)
            
            # More lenient filtering with adaptive thresholds
            if area > min_area and 0.2 < aspect_ratio < 15:
                # Expand region slightly to ensure full capture
                x = max(0, x - 5)
                y = max(0, y - 5)
                w = min(width - x, w + 10)
                h = min(height - y, h + 10)
                math_regions.append((x, y, w, h))
        
        # Merge overlapping regions
        math_regions = self._merge_overlapping_regions(math_regions)
        return math_regions

    def _merge_overlapping_regions(self, regions):
        """Helper method to merge overlapping regions"""
        if not regions:
            return regions
            
        # Sort regions by x coordinate
        regions = sorted(regions, key=lambda x: x[0])
        merged = []
        current = list(regions[0])
        
        for next_region in regions[1:]:
            # Check if regions overlap
            if (current[0] <= next_region[0] <= current[0] + current[2] or
                next_region[0] <= current[0] <= next_region[0] + next_region[2]):
                # Merge the regions
                x_min = min(current[0], next_region[0])
                y_min = min(current[1], next_region[1])
                x_max = max(current[0] + current[2], next_region[0] + next_region[2])
                y_max = max(current[1] + current[3], next_region[1] + next_region[3])
                current = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                merged.append(tuple(current))
                current = list(next_region)
        
        merged.append(tuple(current))
        return merged

    def extract_math_expressions(self, image_path):
        """
        Extract mathematical expressions from the image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            list: List of dictionaries containing extracted math expressions and their LaTeX representations
        """
        try:
            # Preprocess the image
            preprocessed = self.preprocess_image(image_path)
            
            # Detect math regions
            math_regions = self.detect_math_regions(preprocessed)
            
            # Original image for LaTeX OCR
            original_image = Image.open(image_path).convert('RGB')
            
            results = []
            for i, (x, y, w, h) in enumerate(math_regions):
                try:
                    # Extract region
                    region = preprocessed[y:y+h, x:x+w]
                    
                    # Convert region to PIL Image for text extraction
                    region_pil = Image.fromarray(region)
                    
                    # Get original region for LaTeX extraction
                    original_region = original_image.crop((x, y, x+w, y+h))
                    
                    # Extract text using Tesseract with math-specific configuration
                    text = pytesseract.image_to_string(
                        region_pil, 
                        config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789+-*/()=xyz'
                    )
                    
                    latex = ''
                    confidence = 0.0
                    
                    if self.latex_ocr:
                        try:
                            # Preprocess specifically for LaTeX
                            latex_optimized = self.preprocess_for_latex(original_region)
                            
                            # Extract LaTeX using pix2tex
                            try:
                                with torch.no_grad():  # Ensure no gradients are computed
                                    # Convert image to tensor
                                    latex_result = self.latex_ocr(latex_optimized)
                                    
                                    # Handle different output types
                                    if isinstance(latex_result, (list, tuple, np.ndarray)):
                                        # Convert numpy array to list if needed
                                        if isinstance(latex_result, np.ndarray):
                                            latex_result = latex_result.tolist()
                                        latex = str(latex_result[0]) if latex_result else ''
                                    elif isinstance(latex_result, torch.Tensor):
                                        # Convert tensor to list
                                        latex_result = latex_result.detach().cpu().numpy().tolist()
                                        latex = str(latex_result[0]) if latex_result else ''
                                    else:
                                        latex = str(latex_result)
                                    
                                    # Clean up LaTeX string
                                    latex = latex.strip()
                                    if not latex:
                                        latex = ''
                                    
                                    # Add basic LaTeX formatting if needed
                                    if latex and not latex.startswith('\\'):
                                        latex = '\\' + latex
                                    
                                    # Basic confidence scoring based on LaTeX syntax
                                    confidence = self._calculate_latex_confidence(latex)
                                    
                            except Exception as model_err:
                                print(f"LaTeX model error for region {i+1}: {str(model_err)}")
                                latex = ''
                                confidence = 0.0
                            
                        except Exception as latex_err:
                            print(f"LaTeX extraction error for region {i+1}: {str(latex_err)}")
                            latex = ''
                            confidence = 0.0
                    
                    # Ensure all values are Python native types
                    result = {
                        'region_id': int(i + 1),
                        'text': str(text).strip(),
                        'latex': str(latex).strip() if latex else '',
                        'confidence': float(confidence),
                        'bbox': tuple(map(int, (x, y, w, h)))  # Convert bbox values to integers
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing region {i+1}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"Error extracting math expressions: {str(e)}")
            return []

    def _calculate_latex_confidence(self, latex):
        """
        Calculate a basic confidence score for extracted LaTeX
        
        Args:
            latex (str): Extracted LaTeX string
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not latex:
            return 0.0
            
        # Check for balanced brackets and common math operators
        score = 0.0
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        # Common mathematical LaTeX commands
        math_commands = ['\\frac', '\\sqrt', '\\sum', '\\int', '\\lim', 
                        '\\alpha', '\\beta', '\\theta', '\\pi', '\\infty']
                        
        # Check bracket balance
        for char in latex:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return 0.0  # Unbalanced brackets
                if char != brackets[stack.pop()]:
                    return 0.0  # Mismatched brackets
        
        if stack:  # Unclosed brackets
            return 0.0
            
        # Add points for common math operators
        score += min(0.3, 0.05 * sum(1 for op in ['+', '-', '*', '/', '='] if op in latex))
        
        # Add points for LaTeX commands
        score += min(0.4, 0.1 * sum(1 for cmd in math_commands if cmd in latex))
        
        # Add points for length and complexity
        score += min(0.3, len(latex) / 100)  # Longer expressions are more likely to be valid
        
        return min(1.0, score + 0.2)  # Base confidence of 0.2

    def visualize_results(self, image_path, results):
        """
        Visualize detected math regions on the image
        
        Args:
            image_path (str): Path to the input image
            results (list): List of detected math expressions and their regions
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Draw rectangles around detected regions
            for result in results:
                x, y, w, h = result['bbox']
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"Region {result['region_id']}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
            
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create a new figure with specified size
            plt.figure(figsize=(15, 10))
            plt.imshow(image_rgb)
            plt.axis('off')
            
            # Save the visualization
            output_path = os.path.join(os.path.dirname(image_path), 'visualization.png')
            plt.savefig(output_path)
            plt.close()
            
            print(f"Visualization saved to: {output_path}")
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")

def main():
    """Main function to run the MathScrape application"""
    print("Welcome to MathScrape!")
    print("This application extracts mathematical expressions from images.")
    
    while True:
        # Get image path from user
        image_path = input("\nEnter the path to your image (or 'quit' to exit): ")
        
        if image_path.lower() == 'quit':
            print("\nThank you for using MathScrape! Goodbye!")
            break
        
        try:
            # Create MathScrape instance
            scraper = MathScrape()
            
            # Process image
            results = scraper.extract_math_expressions(image_path)
            
            # Display results
            print("\nExtracted Mathematical Expressions:")
            print("-" * 40)
            for result in results:
                print(f"\nRegion {result['region_id']}:")
                print(f"Text: {result['text']}")
                print(f"LaTeX: {result['latex']}")
                print(f"Confidence: {result['confidence']:.2f}")
            
            # Visualize results
            scraper.visualize_results(image_path, results)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please make sure the image path is correct and try again.")

if __name__ == "__main__":
    main()
