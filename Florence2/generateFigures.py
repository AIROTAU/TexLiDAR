import os
import matplotlib.pyplot as plt
from PIL import Image
import textwrap
import numpy as np


image_folder = r"C:\temp\S"
text_file = r"C:\temp\S\DETAILED_CAPTION.txt"
output_folder = r'C:\temp\S\DETAILED_CAPTION'


# Check if the output folder exists, and create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the text file containing image names and descriptions
with open(text_file, 'r') as file:
    lines = file.readlines()

# Loop over each line
for line in lines:
    # Split the line into filename and caption
    filename, caption = line.split(':', 1)
    filename = filename.strip()  # Remove any extra spaces
    caption = caption.strip()    # Remove any extra spaces

    # Get the full image path
    img_path = os.path.join(image_folder, filename)

    # Check if the image file exists
    if os.path.exists(img_path):
        # Read the image using PIL
        img = Image.open(img_path)

        # Create a figure
        plt.figure(figsize=(10, 10))

        # If the image is grayscale, use the 'gray' colormap
        # Assuming img is the PIL image object
        if img.mode == 'L':  # 'L' means 8-bit grayscale (uint8)
            plt.imshow(img, cmap='gray')
        elif img.mode == 'I;16':  # 'I;16' means 16-bit grayscale (uint16)
            # Convert the image to a numpy array and normalize to [0, 1]
            img = np.asarray(img) / 65535.0
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        plt.axis('off')  # Hide the axis

        # Use textwrap to wrap the caption while respecting word boundaries
        wrapped_caption = textwrap.fill(caption, width=100)  # Adjust width as needed

        # Set the title with the wrapped caption and smaller font size
        plt.title(wrapped_caption, fontsize=10, pad=20)

        # Save the image with the caption in the title
        output_path = os.path.join(output_folder, f"{filename}.png")
        plt.savefig(output_path, bbox_inches='tight')

        # Close the figure
        plt.close()
    else:
        print(f"Image file not found: {img_path}")
