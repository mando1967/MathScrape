import os
# Set OpenMP environment variable before importing any other packages
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
from pix2tex.cli import LatexOCR
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import platform

class MathScrape:
    def __init__(self):
        """Initialize MathScrape with OCR and LaTeX recognition capabilities"""
        # Initialize OCR
        if platform.system() == 'Windows':
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if not os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
                raise RuntimeError(
                    "Tesseract not found. Please ensure it is installed at 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'"
                )
        
        # Configuration parameters
        self.horizontal_tolerance = 0.2  # 20% of image width
        self.vertical_tolerance = 0.8  # 80% of region height
        
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
        
    def _preprocess_image(self, image_path):
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
            list: List of bounding boxes for detected math regions, grouped by problem
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
        horizontal_merge_distance = width * self.horizontal_tolerance  # Use image width for horizontal tolerance
        
        # Filter and expand contours
        math_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h)
            
            if area > min_area and 0.2 < aspect_ratio < 15:
                # Expand region more horizontally than vertically
                x = max(0, x - 15)  # More padding on left
                y = max(0, y - 5)
                w = min(width - x, w + 30)  # More padding on right
                h = min(height - y, h + 10)
                math_regions.append((x, y, w, h))
        
        if not math_regions:
            return []
            
        # Sort by vertical position
        math_regions.sort(key=lambda x: x[1])
        
        # First pass: merge horizontally close regions at similar vertical positions
        merged_horizontally = []
        current_line = [math_regions[0]]
        
        for region in math_regions[1:]:
            last_region = current_line[-1]
            vertical_diff = abs(region[1] - last_region[1])
            horizontal_gap = region[0] - (last_region[0] + last_region[2])
            
            # More lenient horizontal merging using image-based tolerance
            if vertical_diff < max(last_region[3], region[3]) * 0.5 and horizontal_gap < horizontal_merge_distance:
                current_line.append(region)
            else:
                # Merge current line into a single region
                if current_line:
                    x_min = min(r[0] for r in current_line)
                    y_min = min(r[1] for r in current_line)
                    x_max = max(r[0] + r[2] for r in current_line)
                    y_max = max(r[1] + r[3] for r in current_line)
                    # Add extra padding to merged regions
                    x_min = max(0, x_min - 10)
                    x_max = min(width, x_max + 10)
                    merged_horizontally.append((x_min, y_min, x_max - x_min, y_max - y_min))
                current_line = [region]
        
        # Don't forget the last line
        if current_line:
            x_min = min(r[0] for r in current_line)
            y_min = min(r[1] for r in current_line)
            x_max = max(r[0] + r[2] for r in current_line)
            y_max = max(r[1] + r[3] for r in current_line)
            # Add extra padding to merged regions
            x_min = max(0, x_min - 10)
            x_max = min(width, x_max + 10)
            merged_horizontally.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        # Second pass: group into problems based on vertical spacing
        merged_horizontally.sort(key=lambda x: x[1])
        problem_groups = []
        current_problem = [merged_horizontally[0]]
        
        for region in merged_horizontally[1:]:
            last_region = current_problem[-1]
            vertical_gap = region[1] - (last_region[1] + last_region[3])
            
            # Use vertical tolerance based on region height
            if vertical_gap > last_region[3] * self.vertical_tolerance:
                problem_groups.append(current_problem)
                current_problem = [region]
            else:
                current_problem.append(region)
        
        if current_problem:
            problem_groups.append(current_problem)
        
        return problem_groups

    def _merge_overlapping_regions(self, regions):
        """Helper method to merge overlapping regions"""
        if not regions:
            return regions
        
        # For a single problem, merge all regions into one
        x_min = min(r[0] for r in regions)
        y_min = min(r[1] for r in regions)
        x_max = max(r[0] + r[2] for r in regions)
        y_max = max(r[1] + r[3] for r in regions)
        
        return [(x_min, y_min, x_max - x_min, y_max - y_min)]

    def extract_math_expressions(self, image_path):
        """
        Extract mathematical expressions from the image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            list: List of dictionaries containing extracted expressions and their bounding boxes
        """
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed = self._preprocess_image(image_path)
        
        # Detect math regions
        problem_groups = self.detect_math_regions(preprocessed)
        
        results = []
        problem_id = 1
        
        for group in problem_groups:
            # Merge all regions in the group into one
            x_min = min(r[0] for r in group)
            y_min = min(r[1] for r in group)
            x_max = max(r[0] + r[2] for r in group)
            y_max = max(r[1] + r[3] for r in group)
            merged_region = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Extract text from the merged region
            x, y, w, h = merged_region
            roi = gray[y:y+h, x:x+w]
            
            # Ensure minimum size for OCR
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                continue
                
            # Extract text using Tesseract
            text = pytesseract.image_to_string(
                roi,
                config='--psm 7 -c tessedit_char_whitelist=0123456789+-=()[]{}^*/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ∫∞→'
            ).strip()
            
            # Debug print to see exact characters
            print(f"Raw OCR output: {[ord(c) for c in text]}")
            print(f"Characters: {[c for c in text]}")
            
            # Preprocess text for LaTeX conversion
            processed_text = text
            
            # Handle polynomial terms
            import re
            
            # Replace x? with x^2 (common OCR misrecognition)
            processed_text = processed_text.replace('x?', 'x^2')
            
            # Handle other polynomial patterns
            processed_text = re.sub(r'x(\d)', r'x^{\1}', processed_text)  # Convert x2 to x^{2}
            processed_text = re.sub(r'(\d)x', r'\1x', processed_text)     # Ensure proper coefficient format
            
            # Replace common x-like characters with 'x'
            x_like_chars = ['×', '✕', '⨯', 'х', 'Х']  # Various Unicode x-like characters
            for x_char in x_like_chars:
                processed_text = processed_text.replace(x_char, 'x')
            
            # Handle integral symbol (including when recognized as 'J')
            if 'J' in text or '∫' in text:
                # Replace 'J' or '∫' with proper LaTeX integral
                processed_text = processed_text.replace('J', '\\int')
                processed_text = processed_text.replace('∫', '\\int')
                
                # Handle polynomial expressions within integrals
                import re
                
                # Handle x^2 patterns (including when recognized as x2)
                processed_text = re.sub(r'x2', 'x^2', processed_text)
                processed_text = re.sub(r'x\^2', 'x^{2}', processed_text)
                
                # Handle addition/subtraction in polynomials
                processed_text = processed_text.replace('-+', '-')  # Fix common OCR error
                
                # Add multiplication symbol for coefficient terms
                processed_text = re.sub(r'(\d)x', r'\1\\cdot x', processed_text)
                
                # Handle dx at the end of integral
                processed_text = re.sub(r'dx$', '\\,dx', processed_text)
                
                # Handle common integral patterns
                integral_patterns = {
                    r'\\int(\d+)': r'\\int_{0}^{\\1}',  # Basic definite integral
                    r'\\int_(\d+)': r'\\int_{\\1}',      # Lower bound
                    r'\\int\^(\d+)': r'\\int^{\\1}',      # Upper bound
                    r'\\int_(\d+)\^(\d+)': r'\\int_{\\1}^{\\2}'  # Both bounds
                }
                
                for pattern, replacement in integral_patterns.items():
                    processed_text = re.sub(pattern, replacement, processed_text)
            
            # Handle limit notation patterns
            if 'lim' in text:
                # Common limit patterns
                limit_patterns = {
                    'lim(x--)': r'\lim_{x \to \infty}',
                    'lim(x->)': r'\lim_{x \to \infty}',
                    'lim(x->∞)': r'\lim_{x \to \infty}',
                    'lim(x->0)': r'\lim_{x \to 0}',
                    'lim(x->-∞)': r'\lim_{x \to -\infty}',
                    'lim(n--)': r'\lim_{n \to \infty}',
                    'lim(n->∞)': r'\lim_{n \to \infty}'
                }
                
                for pattern, replacement in limit_patterns.items():
                    if pattern in processed_text:
                        processed_text = processed_text.replace(pattern, replacement)
                        break
                
                # Handle general patterns
                processed_text = processed_text.replace('lim(x-', r'\lim_{x \to')
                processed_text = processed_text.replace('lim(n-', r'\lim_{n \to')
                processed_text = processed_text.replace('--)', r'\infty}')
                processed_text = processed_text.replace('->)', r'\infty}')
            
            # Handle fractions
            if '/' in processed_text:
                # Match patterns like "a/b" where a and b can be numbers or variables
                import re
                processed_text = re.sub(r'(\d+|[a-zA-Z])/(\d+|[a-zA-Z])', r'\\frac{\1}{\2}', processed_text)
            
            # Handle other common mathematical notations
            common_replacements = {
                '->': r'\to',
                '^2': r'^{2}',
                '^3': r'^{3}',
                '>=': r'\geq',
                '<=': r'\leq',
                '!=': r'\neq',
                '∞': r'\infty',
                'pi': r'\pi',
                'theta': r'\theta',
                'alpha': r'\alpha',
                'beta': r'\beta',
                'sqrt': r'\sqrt'
            }
            
            for pattern, replacement in common_replacements.items():
                processed_text = processed_text.replace(pattern, replacement)
            
            # Extract LaTeX if text was found
            latex = None
            if text and self.latex_ocr is not None:
                try:
                    # Convert numpy array to PIL Image for LaTeX OCR
                    pil_roi = Image.fromarray(roi)
                    # Try using processed text if available
                    if processed_text != text:
                        latex = processed_text
                    else:
                        latex = self.latex_ocr(pil_roi)
                except Exception as e:
                    print(f"LaTeX extraction error: {e}")
            
            if text or latex:
                results.append({
                    'problem_id': problem_id,
                    'text': text,
                    'latex': latex,
                    'bbox': merged_region
                })
            
            problem_id += 1
        
        # Print results
        print("\nExtracted Mathematical Expressions:")
        print("----------------------------------------\n")
        
        current_problem = None
        for result in results:
            if current_problem != result['problem_id']:
                if current_problem is not None:
                    print()
                current_problem = result['problem_id']
                print(f"Problem {result['problem_id']}:")
            
            print(f"Text: {result['text']}")
            if result['latex']:
                print(f"LaTeX: {result['latex']}")
        
        # Visualize results
        self.visualize_results(image, results)
        
        return results

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

    def visualize_results(self, image, results):
        """
        Visualize detected math regions on the image
        
        Args:
            image (numpy.ndarray): Input image
            results (list): List of detected math expressions and their regions
        """
        # Create figure and axis
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Define colors for different problems
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        
        # Draw rectangles around detected regions
        current_ax = plt.gca()
        for result in results:
            problem_id = result['problem_id']
            x, y, w, h = result['bbox']
            color = colors[(problem_id - 1) % len(colors)]
            
            # Draw rectangle
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            current_ax.add_patch(rect)
            
            # Add problem number
            plt.text(x, y-5, f'Problem {problem_id}', color=color, fontsize=10)
        
        plt.axis('off')
        plt.savefig('visualization.png')
        plt.close()

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
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please make sure the image path is correct and try again.")

if __name__ == "__main__":
    main()
