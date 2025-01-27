import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_WIDTH = 2048  # Full horizontal resolution of the LiDAR scan
CROP_WIDTH = 512  # Width of the cropped area
OFFSET = 256  # Horizontal offset for each area

def adjust_roi_in_full_image(roi, direction):
    """
    Adjust the given ROI for a specific direction in the full image based on direction.

    Parameters:
    - roi: Tuple (x_min, y_min, x_max, y_max) for the ROI in the cropped slice.
    - direction: One of ['left', 'front', 'right', 'back'] indicating the direction.

    Returns:
    - Tuple (x_min_full, y_min, x_max_full, y_max) for the adjusted ROI in the full image.
    """

    # Map direction to slice index
    direction_map = {
        'left': 1,
        'front': 2,
        'right': 3,
        'back': 4
    }

    # Ensure the direction is valid
    if direction not in direction_map:
        raise ValueError(f"Invalid direction. Choose from ['left', 'front', 'right', 'back'].")

    # Get the slice index for the given direction
    index = direction_map[direction]

    # Calculate start column for the slice in the full image
    start_col = (index - 1) * CROP_WIDTH + OFFSET

    # Adjust the x-coordinates in the ROI for the full image
    x_min_crop, y_min, x_max_crop, y_max = roi
    x_min_full = (x_min_crop + start_col) % IMAGE_WIDTH
    x_max_full = (x_max_crop + start_col) % IMAGE_WIDTH

    return x_min_full, y_min, x_max_full, y_max


def plot_image_with_roi(image_path, roi, direction):
    """
    Plot the image and overlay the adjusted ROI for a given direction.

    Parameters:
    - image_path: Path to the image file.
    - roi: List of tuples for ROIs in the full image.
    - direction: One of ['left', 'front', 'right', 'back'] indicating the direction.
    """
    # Load the image
    image = np.array(Image.open(image_path))

    # Plot the image
    plt.imshow(image, cmap='gray')
    plt.title(f"Image with ROI for direction: {direction}")

    # Plot the rectangles for the ROIs
    for r in roi:
        x_min_full, y_min, x_max_full, y_max = r
        plt.gca().add_patch(plt.Rectangle((x_min_full, y_min), x_max_full - x_min_full, y_max - y_min,
                                          linewidth=2, edgecolor='red', facecolor='none'))

    # Show the plot
    plt.show()


# Main function
if __name__ == "__main__":
    roi = (71.93600463867188, 63.552001953125, 99.5840072631836, 109.24800872802734)
    roi = tuple(map(round, roi))
    image_path =  r"C:\TexLiDAR\Images\8344.png"
    direction = 'front'

    # Adjust the ROI for the full image
    x_min_full, y_min, x_max_full, y_max = adjust_roi_in_full_image(roi, direction)

    # Handle wraparound for "back" slice
    if x_min_full > x_max_full:
        # If the ROI crosses the image boundary, split it into two rectangles
        roi = [(x_min_full, y_min, IMAGE_WIDTH - 1, y_max), (0, y_min, x_max_full, y_max)]
    else:
        # Single ROI without wraparound
        roi = [(x_min_full, y_min, x_max_full, y_max)]

    # Plot the image with the ROI
    plot_image_with_roi(image_path, roi, direction)
