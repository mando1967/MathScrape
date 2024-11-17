from mathscrape import MathScrape
import os

def main():
    # Initialize MathScrape
    print("Initializing MathScrape...")
    scraper = MathScrape()
    
    # Get the test image path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(current_dir, "test_math_problems.png")
    
    if not os.path.exists(test_image):
        print(f"Test image not found at: {test_image}")
        return
        
    print(f"\nProcessing image: {test_image}")
    try:
        # Extract math expressions
        results = scraper.extract_math_expressions(test_image)
        
        # Print results
        print("\nExtracted Math Expressions:")
        print("-" * 50)
        for result in results:
            print(f"\nRegion {result['region_id']}:")
            print(f"Text: {result['text']}")
            print(f"LaTeX: {result['latex']}")
            print(f"Confidence: {result['confidence']:.2f}")
        
        # Visualize results
        scraper.visualize_results(test_image, results)
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
