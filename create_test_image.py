import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    # Create a white image
    width = 800
    height = 400
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use Arial font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    # Draw some math problems
    problems = [
        "x² + 5x + 6 = 0",
        "∫(x² + 1)dx",
        "lim(x→∞) 1/x = 0"
    ]
    
    y_position = 50
    for problem in problems:
        # Center the text horizontally
        text_width = draw.textlength(problem, font=font)
        x_position = (width - text_width) // 2
        
        # Draw the problem
        draw.text((x_position, y_position), problem, fill='black', font=font)
        y_position += 100
    
    # Save the image
    image.save("test_math_problems.png")
    print(f"Test image created: {os.path.abspath('test_math_problems.png')}")

if __name__ == "__main__":
    create_test_image()
