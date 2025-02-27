# Configuration file for constants

# Image alignment constants
IMAGE_WIDTH = 2048  # Full image width
IMAGE_HEIGHT = 128  # Full image height
CROP_WIDTH = 512    # Width of the cropped area
OFFSET = 256        # Horizontal offset for each section

# Mapping directions to slice index
direction_map = {
    'left': 1,
    'front': 2,
    'right': 3,
    'back': 4
}

directions = list(direction_map.keys())

# Prompt pool for caption generation
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
