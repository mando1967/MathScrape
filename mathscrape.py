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
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, AgglomerativeClustering

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
        self.horizontal_tolerance = 0.8  # 80% of image width - increased to group characters into expressions
        self.vertical_tolerance = 0.6    # 60% of region height
        self.min_area = 50             # Minimum area in pixels
        self.max_gap = 30               # Maximum gap between components - increased for character spacing
        
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
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size
            2    # C constant
        )
        
        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Dilate to connect components
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        return binary

    def preprocess_for_latex(self, image, attempt=0):
        """
        Special preprocessing for LaTeX recognition
        
        Args:
            image (PIL.Image): Input image
            attempt (int): Attempt number to try different preprocessing parameters
        
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
        
        # Different preprocessing strategies based on attempt number
        if attempt == 0:
            # Default preprocessing - good for general cases
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img_array)
            denoised = cv2.fastNlMeansDenoising(enhanced)
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        elif attempt == 1:
            # Enhanced contrast and morphology - better for symbols like infinity
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            enhanced = clahe.apply(img_array)
            
            # Apply morphological operations to connect broken symbols
            kernel = np.ones((2,2), np.uint8)
            dilated = cv2.dilate(enhanced, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            # Use adaptive thresholding for better local contrast
            binary = cv2.adaptiveThreshold(
                eroded,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
    
        else:
            # Focus on thin strokes and details - better for differentials
            # Sharpen the image first
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(img_array, -1, kernel)
            
            # Apply bilateral filter to preserve edges while removing noise
            smoothed = cv2.bilateralFilter(sharpened, 9, 75, 75)
            
            # Use Otsu's thresholding on the smoothed image
            _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Thin operation to clean up strokes
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.erode(binary, kernel, iterations=1)
    
        # Invert if background is dark
        if np.mean(binary[0:10, 0:10]) < 127:
            binary = cv2.bitwise_not(binary)
    
        # Convert back to PIL Image in RGB mode (pix2tex expects RGB)
        rgb_array = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
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
        Detect regions containing mathematical expressions using advanced clustering
        and dynamic spacing analysis.
        
        Args:
            preprocessed_image (numpy.ndarray): Preprocessed image
            
        Returns:
            list: List of bounding boxes for detected math regions
        """
        # Save preprocessed image for debugging
        cv2.imwrite('debug_preprocessed.png', preprocessed_image)
        
        # Find contours
        contours, _ = cv2.findContours(
            preprocessed_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
            
        # Get all bounding boxes and their centers
        boxes = []
        centers = []
        heights = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            center_x = x + w/2
            center_y = y + h/2
            boxes.append((x, y, w, h, area))
            centers.append((center_x, center_y))
            heights.append(h)
        
        centers = np.array(centers)
        if len(centers) < 2:
            return []
            
        # Calculate nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(centers)
        distances, _ = nbrs.kneighbors(centers)
        
        # Get median character height and spacing
        median_height = np.median(heights)
        char_spacing = np.median(distances[:, 1])
        
        # Calculate more robust vertical spacing using height distribution
        height_std = np.std(heights)
        mean_height = np.mean(heights)
        
        # Get y-coordinate clusters
        y_coords = centers[:, 1].reshape(-1, 1)
        
        # Normalize y-coordinates to handle varying scales
        y_coords_norm = (y_coords - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords))
        
        # Use DBSCAN with adaptive epsilon based on character height
        vertical_eps = (mean_height * 1.5) / (np.max(y_coords) - np.min(y_coords))  # Normalized epsilon
        db = DBSCAN(eps=vertical_eps, min_samples=1).fit(y_coords_norm)
        
        # Get unique lines and their components
        unique_labels = np.unique(db.labels_)
        lines = []
        for label in unique_labels:
            mask = db.labels_ == label
            if np.any(mask):
                # Get all boxes in this cluster
                cluster_boxes = [boxes[i] for i, m in enumerate(mask) if m]
                
                # Calculate cluster bounds
                y_min = min(box[1] for box in cluster_boxes)
                y_max = max(box[1] + box[3] for box in cluster_boxes)
                cluster_center = (y_min + y_max) / 2
                
                lines.append((cluster_center, cluster_boxes))
        
        # Sort lines by y-coordinate
        lines.sort(key=lambda x: x[0])
        
        # Save debug image with detected lines
        debug_img = cv2.cvtColor(preprocessed_image.copy(), cv2.COLOR_GRAY2BGR)
        colors = [(0,255,0), (255,0,0), (0,0,255)]  # Different colors for each line
        
        for i, (center_y, line_boxes) in enumerate(lines):
            color = colors[i % len(colors)]
            # Get full extent of line
            y_min = min(box[1] for box in line_boxes)
            y_max = max(box[1] + box[3] for box in line_boxes)
            
            # Draw boxes and line
            for box in line_boxes:
                x, y, w, h, _ = box
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 1)
            # Draw line through center of cluster
            cv2.line(debug_img, (0, int(center_y)), (debug_img.shape[1], int(center_y)), color, 1)
        
        cv2.imwrite('debug_lines.png', debug_img)
        
        # Process each line to create single bounding box per equation
        grouped_regions = []
        for _, line_boxes in lines:
            if not line_boxes:
                continue
                
            # Create single bounding box for entire line
            x_min = min(box[0] for box in line_boxes)
            y_min = min(box[1] for box in line_boxes)
            x_max = max(box[0] + box[2] for box in line_boxes)
            y_max = max(box[1] + box[3] for box in line_boxes)
            
            # Add padding
            pad_x = int(0.05 * (x_max - x_min))
            pad_y = int(0.4 * (y_max - y_min))
            
            grouped_regions.append((
                max(0, x_min - pad_x),
                max(0, y_min - pad_y),
                min(preprocessed_image.shape[1], x_max - x_min + 2*pad_x),
                min(preprocessed_image.shape[0], y_max - y_min + 2*pad_y)
            ))
        
        # Save debug image with grouped regions
        debug_groups = cv2.cvtColor(preprocessed_image.copy(), cv2.COLOR_GRAY2BGR)
        for x, y, w, h in grouped_regions:
            cv2.rectangle(debug_groups, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite('debug_groups.png', debug_groups)
        
        return grouped_regions

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
        problem_regions = self.detect_math_regions(preprocessed)
        
        results = []
        problem_id = 1
        
        for region in problem_regions:
            # Region is already in (x, y, w, h) format
            x, y, w, h = region
            roi = gray[y:y+h, x:x+w]
            
            # Ensure minimum size for OCR
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                continue
                
            # Extract text from the region
            text = pytesseract.image_to_string(
                roi,
                config='--psm 7 -c tessedit_char_whitelist=0123456789+-=()[]{}^*/\\JIlimabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ∫∞→_≠≤≥∂∑∏√∫∮∇∆∈∉⊂⊃⊆⊇∪∩ '
            ).strip()
            
            # Fix common OCR issues in raw text
            text = re.sub(r'([a-zA-Z])\?', r'\1^2', text)  # Fix x? -> x^2
            text = re.sub(r'\{([a-zA-Z])\}\?', r'{\1}^2', text)  # Fix {x}? -> {x}^2
            
            print(f"\nRegion {problem_id} - DETAILED DEBUG:")
            print(f"Raw OCR text (repr): {repr(text)}")
            print(f"Raw OCR text (hex): {' '.join(hex(ord(c)) for c in text)}")
            
            # Split text into separate expressions if multiple lines detected
            expressions = [expr.strip() for expr in text.split('\n') if expr.strip()]
            print(f"Split expressions: {expressions}")
            
            for expr in expressions:
                print(f"\nProcessing expression:")
                print(f"Raw expr (repr): {repr(expr)}")
                print(f"Raw expr (hex): {' '.join(hex(ord(c)) for c in expr)}")
                
                # Try LaTeX OCR with multiple attempts
                latex = None
                best_latex = None
                best_confidence = 0
                
                if self.latex_ocr:
                    for attempt in range(3):  # Try up to 3 different preprocessing strategies
                        try:
                            # Convert region to PIL Image
                            roi_pil = Image.fromarray(roi)
                            # Preprocess with different parameters
                            roi_pil = self.preprocess_for_latex(roi_pil, attempt)
                            # Get LaTeX
                            latex = self.latex_ocr(roi_pil)
                            print(f"LaTeX OCR attempt {attempt + 1} output (repr): {repr(latex)}")
                            print(f"LaTeX OCR attempt {attempt + 1} output (hex): {' '.join(hex(ord(c)) for c in latex)}")
                            
                            # Clean up and validate LaTeX
                            cleaned_latex = self._cleanup_latex(latex)
                            if cleaned_latex:
                                confidence = self._calculate_latex_confidence(cleaned_latex)
                                if confidence > best_confidence:
                                    best_latex = cleaned_latex
                                    best_confidence = confidence
                                    print(f"New best LaTeX (confidence {confidence:.2f}): {repr(best_latex)}")
                            
                        except Exception as e:
                            print(f"LaTeX OCR attempt {attempt + 1} failed: {str(e)}")
                            continue
                
                # Use the best LaTeX result or fall back to cleaned OCR text
                if best_latex:
                    latex = best_latex
                else:
                    # Try to clean up the OCR text as LaTeX
                    latex = self._cleanup_latex(expr)
                
                if text or latex:
                    results.append({
                        'problem_id': problem_id,
                        'text': expr,
                        'latex': latex if latex else expr,
                        'confidence': best_confidence if latex else 0.0,
                        'bbox': region
                    })
                    print(f"Added result - Text: {repr(expr)}, LaTeX: {repr(latex if latex else expr)}, Confidence: {best_confidence:.2f}")
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
        
        score = 0.2  # Base confidence
        
        # Penalize overly complex expressions
        score -= 0.1 * latex.count('\\mathcal')
        score -= 0.1 * latex.count('\\chi')
        score -= 0.05 * latex.count('\\textstyle')
        
        # Boost score for proper mathematical notation
        if '\\,d ' in latex:  # Proper differential notation
            score += 0.3
        if '\\to\\infty' in latex:  # Proper infinity in limits
            score += 0.3
        if '\\int' in latex and '\\,d' in latex:  # Complete integral with differential
            score += 0.2
        if '\\lim' in latex and '\\to' in latex:  # Complete limit with arrow
            score += 0.2
        
        # Penalize unusual character combinations
        if 'Q\\chi' in latex or '{Q}\\chi' in latex:
            score -= 0.3
        
        # Add points for balanced delimiters
        score += 0.1 * min(
            latex.count('(') + latex.count('[') + latex.count('\\{'),
            latex.count(')') + latex.count(']') + latex.count('\\}')
        )
        
        # Add points for common mathematical structures
        score += 0.1 * len(re.findall(r'\^\{[^}]+\}', latex))  # Proper superscripts
        score += 0.1 * len(re.findall(r'_\{[^}]+\}', latex))   # Proper subscripts
        
        return min(1.0, score)

    def _cleanup_latex(self, latex):
        """
        Clean up common LaTeX OCR issues and validate syntax
        
        Args:
            latex (str): Raw LaTeX string
        
        Returns:
            str: Cleaned LaTeX string or None if invalid
        """
        if not latex:
            return None
        
        # Pre-cleanup: standardize spacing and remove unwanted characters
        latex = latex.strip()
        latex = re.sub(r'\s+', ' ', latex)
        
        # Fix infinity in limit expressions first
        infinity_patterns = [
            (r'\\to\s*[a-z]\^?\{?2\}?', r'\\to\\infty'),  # x\to a^2 -> x\to\infty
            (r'\\to\s*[a-z0-9]+', r'\\to\\infty'),        # x\to a -> x\to\infty
            (r'\\to\s*\\mathcal\{[^}]+\}', r'\\to\\infty'),  # \to\mathcal{X} -> \to\infty
            (r'(?i)infinity', r'\\infty'),
            (r'(?i)inf(?!ty|inity)', r'\\infty'),
            (r'∞', r'\\infty'),
            (r'oo', r'\\infty'),
            (r'[oO]{2}', r'\\infty')
        ]
        
        for pattern, replacement in infinity_patterns:
            latex = re.sub(pattern, replacement, latex)
        
        # Fix differential notation
        differential_patterns = [
            (r'(?<!\\mathrm\{)(?<!\\)d([xyz]|\\?[a-z])', r'\\mathrm{d}\1'),  # dx -> \mathrm{d}x
            (r'\\mathrm\{d\}([xyz])', r'\\mathrm{d}\1'),  # Fix double wrapping
            (r'd([xyz])\}', r'\\mathrm{d}\1}'),  # Fix cases inside braces
            (r'\\,\s*\\mathrm', r'\\,\\mathrm'),  # Fix spacing
            (r'\\mathrm\{d\}\s+([xyz])', r'\\mathrm{d}\1')  # Remove space after d
        ]
        
        for pattern, replacement in differential_patterns:
            latex = re.sub(pattern, replacement, latex)
        
        # Common LaTeX command replacements
        replacements = {
            'lim': r'\lim',
            'sin': r'\sin',
            'cos': r'\cos',
            'tan': r'\tan',
            'int': r'\int',
            'sum': r'\sum',
            'prod': r'\prod',
            'sqrt': r'\sqrt',
            '→': r'\rightarrow',
            '≤': r'\leq',
            '≥': r'\geq',
            '≠': r'\neq',
            '∂': r'\partial',
            '∑': r'\sum',
            '∏': r'\prod',
            '∫': r'\int',
            '∮': r'\oint',
            '∇': r'\nabla',
            '∆': r'\Delta',
            '∈': r'\in',
            '∉': r'\notin',
            '⊂': r'\subset',
            '⊃': r'\supset',
            '⊆': r'\subseteq',
            '⊇': r'\supseteq',
            '∪': r'\cup',
            '∩': r'\cap'
        }
        
        # Apply replacements only if not already in LaTeX format
        for old, new in replacements.items():
            if old in latex and new not in latex:
                latex = latex.replace(old, new)
    
        # Fix integral notation
        latex = re.sub(r'\\int\s*([^d]*?)\s*\\mathrm', r'\\int \1\\,\\mathrm', latex)
    
        # Fix superscripts and subscripts
        latex = re.sub(r'(\d+|\w+|\})\^(\d+|\w+|\{[^}]+\})', r'{\1}^{\2}', latex)
        latex = re.sub(r'(\w+)_(\w+)', r'{\1}_{\2}', latex)
        
        # Fix question mark as superscript 2
        latex = re.sub(r'([a-zA-Z])\?', r'\1^2', latex)
        latex = re.sub(r'\{([a-zA-Z])\}\?', r'{\1}^2', latex)
        
        # Handle nested expressions
        latex = re.sub(r'\^\{(\d+)\}\^\{(\d+)\}', r'^{\1^{\2}}', latex)
    
        # Ensure proper bracketing
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for char in latex:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or char != brackets[stack.pop()]:
                    return None
        if stack:
            return None
    
        # Final cleanup
        # Remove unnecessary \\textstyle
        latex = latex.replace('\\textstyle', '')
    
        # Fix multiple backslashes
        latex = re.sub(r'\\+([a-zA-Z])', r'\\\1', latex)
    
        # Fix spacing around operators
        latex = re.sub(r'\s*([+\-=×÷])\s*', r'\1', latex)
    
        # Clean up any remaining issues
        latex = re.sub(r'\\mathrm\{\\mathrm\{(.*?)\}\}', r'\\mathrm{\1}', latex)  # Fix double mathrm
        latex = re.sub(r'\\infty\\infty', r'\\infty', latex)  # Fix double infinity
        latex = re.sub(r'Q\\chi', r'dx', latex)  # Fix common misrecognition
        latex = re.sub(r'\{\\mathcal\{Q\}\}\\chi', r'dx', latex)  # Fix another common misrecognition
    
        return latex

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
