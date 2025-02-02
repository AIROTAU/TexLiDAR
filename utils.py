import numpy as np
from PIL import Image
import random
import re
from PIL import ImageDraw, ImageFont

# Constants for image alignment
IMAGE_WIDTH = 2048  # Full image width
CROP_WIDTH = 512  # Width of the cropped area
OFFSET = 256  # Horizontal offset for each area

# Mapping directions to slice index
direction_map = {
    'left': 1,
    'front': 2,
    'right': 3,
    'back': 4
}


def adjust_roi_in_full_image(roi, direction):
    """Adjusts an ROI from a section back to full image coordinates."""
    x_min_crop, y_min, x_max_crop, y_max = roi

    if direction not in direction_map:
        raise ValueError(f"Invalid direction. Choose from {list(direction_map.keys())}.")

    # Calculate start column for the slice in the full image
    index = direction_map[direction]
    start_col = (index - 1) * CROP_WIDTH + OFFSET

    # Adjust x-coordinates for full image
    x_min_full = (x_min_crop + start_col) % IMAGE_WIDTH
    x_max_full = (x_max_crop + start_col) % IMAGE_WIDTH

    # Handle wrap-around case for the 'back' section
    if direction == 'back' and x_min_full > x_max_full:
        return [(x_min_full, y_min, IMAGE_WIDTH, y_max),  # First part (right side)
                (0, y_min, x_max_full, y_max)]  # Second part (left side)

    return [(x_min_full, y_min, x_max_full, y_max)]


def draw_bboxes(image, rois, labels):
    """Draws bounding boxes with labels on the image, each in a different color."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Define a set of distinct colors
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "pink"]

    for i, (roi, label) in enumerate(zip(rois, labels)):
        x1, y1, x2, y2 = roi
        color = colors[i % len(colors)]  # Cycle through colors

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Compute text size and draw label background
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1, x1 + text_width + 4, y1 + text_height + 4], fill=color)

        # Draw text
        draw.text((x1 + 2, y1 + 2), label, fill="white", font=font)

    return image


directions = ["left", "front", "right", "back"]

prompt_pool = [
    "The view from the {direction} shows",
    "From the {direction} angle, you can see",
    "Looking towards the {direction}, we see",
    "Seen from the {direction}, the image features",
    "When viewed from the {direction}, the scene depicts",
    "Facing the {direction}, the image highlights",
    "Captured from the {direction}, this image shows",
    "The {direction} perspective reveals",
    "From the {direction} perspective, we observe",
    "As seen from the {direction}, the image presents",
    "From the {direction} viewpoint, we can observe",
    "Looking in the {direction}, the image portrays",
    "Captured from the {direction} angle, the scene depicts",
    "Seen from the {direction} perspective, we see",
    "The {direction} view reveals",
    "From the {direction} side, the image shows",
    "Through the {direction} lens, we observe",
    "The scene viewed from the {direction} highlights",
    "With the {direction}-facing angle, we can see",
    "In the {direction} direction, the image presents",
    "The {direction}-facing perspective showcases",
    "Captured from the {direction}-ward view, we see",
    "When viewed from the {direction}, this image reveals",
    "Peering towards the {direction}, the image shows",
    "In the {direction} direction, the photo captures",
    "The {direction} shot reveals",
    "As viewed from the {direction}, the scene portrays",
    "The {direction} view offers a glimpse of",
    "The {direction}-oriented perspective showcases",
    "From the {direction}-facing view, we can observe"
]


# Function to clean the caption by removing unwanted prefixes
def clean_caption_prefix(captions):
    cleaned_captions = []

    # Define the patterns to remove
    patterns_to_remove = [
        r"The image shows",
        r"a black and white photo of",
        r"([^.]*\bthe image is in black and white\b[^.]*\.)",
        r"([^.]*\bThe image is in black and white\b[^.]*\.)",
    ]

    for caption in captions:
        caption = caption.replace("</s><s>", " ").replace("<s>", "").replace("</s>", "")
        for pattern in patterns_to_remove:
            caption = re.sub(pattern, "", caption)  # Remove the pattern using regex substitution
        cleaned_captions.append(caption.strip())  # Strip any leading/trailing spaces

    return cleaned_captions


# Function to clean and add directions with dynamic prompts
def clean_and_add_direction(captions, directions, prompt_pool):
    # Clean the captions by removing the unwanted prefixes
    captions = clean_caption_prefix(captions)

    cleaned_captions = []
    for i, caption in enumerate(captions):
        # Randomly select a unique prompt from the pool
        prompt = random.choice(prompt_pool)
        prompt_pool.remove(prompt)  # Avoid repeating prompts

        # Add the cleaned caption with a dynamic prompt and direction
        cleaned_caption = f"{prompt.format(direction=directions[i])} {caption}"

        # Append the cleaned caption
        cleaned_captions.append(cleaned_caption)

    return cleaned_captions


# Function to merge the captions for all directions
def merge_captions(captions):
    """Merges the captions for the 4 directions into one."""
    if len(captions) != 4:
        raise ValueError("Expected exactly 4 captions for all directions")

    # Copy the prompt_pool to avoid modifying the original one
    cleaned_captions = clean_and_add_direction(captions, directions, prompt_pool.copy())
    merged_caption = " ".join(cleaned_captions)

    return merged_caption


def split_image_into_sections(image):
    """
    Splits an image into four sections

    Args:
        image(str): PIL image file.

    Returns:
        tuple: Four NumPy arrays representing (left, front, right, back) sections.
    """

    # Convert image to NumPy array
    image_np = np.array(image)

    # Get image dimensions
    height, width, _ = image_np.shape

    # Define small image size and shift amount
    small_image_size = CROP_WIDTH
    shift_amount = OFFSET

    # Ensure width is a multiple of 512
    if width % small_image_size != 0:
        raise ValueError("Image width must be a multiple of 512.")

    # Compute the column start and end dynamically
    def get_section(idx):
        col_start = (idx * small_image_size + shift_amount) % width
        col_end = ((idx + 1) * small_image_size + shift_amount) % width
        if col_end < col_start:
            return np.hstack((image_np[:, col_start:], image_np[:, :col_end]))  # Wrap-around
        return image_np[:, col_start:col_end]

    left = Image.fromarray(get_section(0))
    front = Image.fromarray(get_section(1))
    right = Image.fromarray(get_section(2))
    back = Image.fromarray(get_section(3))

    return left, front, right, back
