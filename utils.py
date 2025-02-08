import numpy as np
import random
import re
from PIL import Image, ImageDraw, ImageFont
from config import IMAGE_WIDTH, CROP_WIDTH, OFFSET, direction_map, prompt_pool, directions


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
        return [(x_min_full, y_min, IMAGE_WIDTH, y_max), (0, y_min, x_max_full, y_max)]

    return [(x_min_full, y_min, x_max_full, y_max)]


def draw_bboxes(image, rois, labels):
    """Draws bounding boxes with labels on the image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "pink"]

    for i, (roi, label) in enumerate(zip(rois, labels)):
        x1, y1, x2, y2 = roi
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1, x1 + text_width + 4, y1 + text_height + 4], fill=color)
        draw.text((x1 + 2, y1 + 2), label, fill="white", font=font)

    return image


def clean_caption_prefix(captions):
    """Cleans captions by removing unwanted prefixes."""
    patterns_to_remove = [re.compile(p) for p in [
        r"The image shows",
        r"a black and white photo of",
        r"([^.]*\bthe image is in black and white\b[^.]*\.)",
        r"([^.]*\bThe image is in black and white\b[^.]*\.)",
    ]]
    cleaned_captions = []
    for caption in captions:
        caption = caption.replace("</s><s>", " ").replace("<s>", "").replace("</s>", "")
        for pattern in patterns_to_remove:
            caption = pattern.sub("", caption)
        cleaned_captions.append(caption.strip())
    return cleaned_captions


def clean_and_add_direction(captions):
    """Cleans captions and adds directional prompts."""
    captions = clean_caption_prefix(captions)
    selected_prompts = random.sample(prompt_pool, len(directions))  # Ensure unique prompts
    return [f"{prompt.format(direction=directions[i])} {captions[i]}" for i, prompt in enumerate(selected_prompts)]


def merge_captions(captions):
    """Merges captions for all directions into one."""
    if len(captions) != len(directions):
        raise ValueError("Mismatch between captions and directions count.")
    return " ".join(clean_and_add_direction(captions))


def split_image_into_sections(image):
    """Splits an RGB image into four sections: (left, front, right, back)."""
    image_np = np.array(image)
    height, width, _ = image_np.shape
    if width % CROP_WIDTH != 0:
        raise ValueError("Image width must be a multiple of 512.")

    def get_section(idx):
        col_start = (idx * CROP_WIDTH + OFFSET) % width
        col_end = ((idx + 1) * CROP_WIDTH + OFFSET) % width
        return (np.hstack((image_np[:, col_start:], image_np[:, :col_end])) if col_end < col_start
                else image_np[:, col_start:col_end])

    return tuple(Image.fromarray(get_section(i)) for i in range(4))


def load_lidar_point_cloud(file_path):
    """Loads a .bin Lidar point cloud file."""
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
