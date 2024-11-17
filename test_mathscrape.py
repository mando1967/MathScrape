from mathscrape import MathScrape
import os

def test_mathscrape():
    print("Starting MathScrape test...")
    
    # Check if test image exists
    image_path = "test_math_problems.png"
    if not os.path.exists(image_path):
        print(f"Error: Test image '{image_path}' not found!")
        return
    
    try:
        # Create MathScrape instance
        print("\nInitializing MathScrape...")
        scraper = MathScrape()
        
        # Process the test image
        print(f"\nProcessing image: {image_path}")
        results = scraper.extract_math_expressions(image_path)
        
        # Display results
        print("\nExtracted Mathematical Expressions:")
        print("-" * 40)
        for result in results:
            print(f"\nRegion {result['region_id']}:")
            print(f"Text: {result['text']}")
            if result['latex']:
                print(f"LaTeX: {result['latex']}")
        
        # Visualize results
        scraper.visualize_results(image_path, results)
        
        print("\nProcessing complete! Check the visualization window.")
        input("Press Enter to exit...")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please ensure:")
        print("1. Tesseract OCR is properly installed")
        print("2. The test image exists and is readable")
        print("3. All required Python packages are installed")

if __name__ == "__main__":
    test_mathscrape()
