import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mplcursors

# Constants for Lidar projections
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 2048
U_STEP = 1  # Horizontal resolution
V_STEP = 1  # Vertical resolution
U_MIN, U_MAX = np.array([0, 360]) / U_STEP  # Horizontal FOV in degrees
V_MIN, V_MAX = np.array([-22.5, 22.5]) / V_STEP  # Vertical FOV in degrees


def adjust_roi_in_full_image(roi, direction):
    CROP_WIDTH = 512  # Width of the cropped area
    OFFSET = 256  # Horizontal offset for each area
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

    # Return the adjusted ROI in the full image
    return x_min_full, y_min, x_max_full, y_max


def calculate_uv_from_image_coords(x_center, y_center, image_width=2048, image_height=128):
    # Image center
    x_mid = image_width / 2
    y_mid = image_height / 2

    # Angular resolution
    u_resolution = 360 / image_width
    v_resolution = 45 / image_height

    # Offsets from the center
    delta_x = x_center - x_mid
    delta_y = -(y_center - y_mid)

    # Calculate U and V
    u_angle = delta_x * u_resolution
    v_angle = delta_y * v_resolution

    return u_angle, v_angle


def load_lidar_point_cloud(file_path):
    """Load a .bin Lidar point cloud file."""
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)


def lidar_to_images(lidar_data):
    """Convert Lidar data to distance, height, and intensity images."""
    x, y, z, intensity = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2], lidar_data[:, 3]
    distance = np.sqrt(x ** 2 + y ** 2)  # Calculate distance

    # Calculate pixel coordinates
    u = np.rint((np.arctan2(x, y) / np.pi * 180 / U_STEP + 90 )).astype(np.int32)
    v = np.rint((-np.arctan2(z, distance) / np.pi * 180 / V_STEP - V_MIN)).astype(np.int32)

    # Initialize images
    width = int(U_MAX - U_MIN + 1)
    height = int(V_MAX - V_MIN + 1)
    distance_image = np.zeros((height, width), dtype=np.float32)
    height_image = np.zeros((height, width), dtype=np.float32)
    intensity_image = np.zeros((height, width), dtype=np.float32)

    # Map attributes to images
    distance_image[v, u] = distance
    height_image[v, u] = z
    intensity_image[v, u] = intensity

    return distance_image, height_image, intensity_image


def plot_images_with_points(distance, height, intensity, points):
    """
    Visualize distance, height, and intensity images with pixel values on hover
    and mark specific points (u, v) on the images.

    Parameters:
    - distance: 2D array for the distance image.
    - height: 2D array for the height image.
    - intensity: 2D array for the intensity image.
    - points: List of (u, v) tuples to mark on the images.
    """
    # Define clipping ranges for visualization (No normalization)
    ranges = {
        "Distance": (distance, 0, 30),
        "Height": (height, 0, 100),
        "Intensity": (intensity, 0, 100),
    }

    # Plot each image
    fig, axes = plt.subplots(3, 1, figsize=(15, 5))
    for ax, (title, (image, c_min, c_max)) in zip(axes, ranges.items()):
        im = ax.imshow(image, cmap="gray", vmin=c_min, vmax=c_max)
        ax.set_title(f"{title} (Clipped to [{c_min}, {c_max}])")
        ax.axis("off")

        # Mark points on the image
        for u, v in points:
            ax.plot(u, v, 'ro')  # Red circle to mark the point
            ax.text(u, v, f"({u},{v})", color='yellow', fontsize=8, ha='right')

        # Add interactive cursor for pixel values
        mplcursors.cursor(im, hover=True)

    plt.tight_layout()
    plt.show()


def calculate_distance_from_roi(distance_image, height_image, intensity_image, roi):
    x_min_full, y_min, x_max_full, y_max = roi

    # If x_min_full > x_max_full, calculate the midpoint
    if x_min_full > x_max_full:
        roi_length = IMAGE_WIDTH - x_min_full + x_max_full
        x_center = (x_min_full + roi_length // 2) % IMAGE_WIDTH
    else:
        x_center = (x_min_full + x_max_full) // 2

    y_center = (y_min + y_max) // 2
    u, v = calculate_uv_from_image_coords(x_center, y_center)
    v = 23 - v
    u = u + 180
    u = round(u)
    v = round(v)
    # Retrieve measurements
    distance = distance_image[v, u]
    height = height_image[v, u]
    intensity = intensity_image[v, u]
    plot_images_with_points(distance_image, height_image, intensity_image, [(u, v)])
    return {
        "distance": distance,
        "height": height,
        "intensity": intensity,
        "x_center": x_center,
        "y_center": y_center
    }


if __name__ == "__main__":
    # Example usage
    FILE_PATH = r"C:\TexLiDAR\Images\8344.bin"
    roi = (71.93600463867188, 63.552001953125, 99.5840072631836, 109.24800872802734)
    DIRECTION = 'front'

    # Adjust the ROI for the full image once here
    adjusted_roi = adjust_roi_in_full_image(roi, DIRECTION)

    # Load Lidar data
    lidar_points = load_lidar_point_cloud(FILE_PATH)

    # Generate distance, height, and intensity images
    distance_img, height_img, intensity_img = lidar_to_images(lidar_points)
    height_img = height_img + 1  # Optional adjustment

    # Calculate measurements from the adjusted ROI
    result = calculate_distance_from_roi(distance_img, height_img, intensity_img, adjusted_roi)

    # Print results
    print(f"Distance: {result['distance']:.2f}")
    print(f"Height: {result['height']:.2f}")
    print(f"Intensity: {result['intensity']:.2f}")
