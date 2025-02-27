"""
This script processes Lidar point cloud data, converts it into distance, height, and intensity maps,
and extracts measurements from a specified region of interest (ROI) on these maps.
It also visualizes the maps and marks the points corresponding to the extracted measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from utils import adjust_roi_in_full_image, load_lidar_point_cloud
import argparse
from config import IMAGE_WIDTH, IMAGE_HEIGHT


def convert_image_coords_to_uv(x_center, y_center, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
    """Converts image coordinates to U, V angular values."""
    u_res, v_res = 360 / image_width, 45 / image_height
    x_mid, y_mid = image_width / 2, image_height / 2
    return (x_center - x_mid) * u_res, -(y_center - y_mid) * v_res


def convert_lidar_data_to_images(lidar_data, u_min=0, u_max=360, v_min=-22.5, v_max=22.5):
    """Converts Lidar data to distance, height, intensity maps."""
    x, y, z, intensity = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2], lidar_data[:, 3]
    distance = np.sqrt(x ** 2 + y ** 2)
    u = np.rint((np.arctan2(x, y) / np.pi * 180 + 90)).astype(np.int32)
    v = np.rint((-np.arctan2(z, distance) / np.pi * 180 - v_min)).astype(np.int32)

    width, height = int(u_max - u_min + 1), int(v_max - v_min + 1)
    distance_map, height_map, intensity_map = np.zeros((height, width)), np.zeros((height, width)), np.zeros(
        (height, width))
    distance_map[v, u], height_map[v, u], intensity_map[v, u] = distance, z, intensity

    return distance_map, height_map, intensity_map


def display_images_with_marked_points(distance_map, height_map, intensity_map, points):
    """Displays maps with points marked."""
    ranges = {"Distance": distance_map, "Height": height_map, "Intensity": intensity_map}
    fig, axes = plt.subplots(3, 1, figsize=(15, 5))
    for ax, (title, image) in zip(axes, ranges.items()):
        im = ax.imshow(image, cmap="gray", vmin=0, vmax=100)
        ax.set_title(f"{title} (Clipped to [0, 100])")
        ax.axis("off")
        for u, v in points:
            ax.plot(u, v, 'ro')
            ax.text(u, v, f"({u},{v})", color='yellow', fontsize=8, ha='right')
        mplcursors.cursor(im, hover=True)
    plt.tight_layout()
    plt.show()


def extract_measurements_from_roi(distance_map, height_map, intensity_map, roi):
    """Extracts measurements from the center of an ROI."""
    if len(roi) == 2:  # Wrap-around case
        x_min_full, y_min, x_max_full, y_max = roi[0][0], roi[0][1], roi[1][2], roi[0][3]
        roi_length = IMAGE_WIDTH - x_min_full + x_max_full
        x_center = (x_min_full + roi_length // 2) % IMAGE_WIDTH
    else:  # Single ROI part
        x_min_full, y_min, x_max_full, y_max = roi[0]
        x_center = (x_min_full + x_max_full) // 2
    y_center = (y_min + y_max) // 2

    u_angle, v_angle = convert_image_coords_to_uv(x_center, y_center)
    v_angle = 23 - v_angle  # Invert v angle for proper orientation
    u_angle = (u_angle + 175) % 360
    u_angle, v_angle = round(u_angle), round(v_angle)
    distance, height, intensity = distance_map[v_angle, u_angle], height_map[v_angle, u_angle], intensity_map[
        v_angle, u_angle]
    display_images_with_marked_points(distance_map, height_map, intensity_map, [(u_angle, v_angle)])

    return {"distance": distance, "height": height, "intensity": intensity, "x_center": x_center, "y_center": y_center}


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Process Lidar data and extract object measurements.")
    parser.add_argument("--file", "-f", type=str, default=r"C:\TexLiDAR\data\point_cloud.bin",
                        help="Path to Lidar file")
    parser.add_argument("--roi", "-r", type=float, nargs=4, default=[71.936, 63.552, 99.584, 109.248],
                        help="ROI coordinates")
    parser.add_argument("--direction", "-d", type=str, choices=["left", "front", "right", "back"], default="front",
                        help="Image section direction")
    args = parser.parse_args()

    # Process the Lidar data
    adjusted_roi = adjust_roi_in_full_image(tuple(args.roi), args.direction)
    lidar_points = load_lidar_point_cloud(args.file)
    distance_map, height_map, intensity_map = convert_lidar_data_to_images(lidar_points)
    height_map += 1  # Optional adjustment

    # Extract and display measurements
    measurement_results = extract_measurements_from_roi(distance_map, height_map, intensity_map, adjusted_roi)
    print(f"Distance: {measurement_results['distance']:.2f}")
    print(f"Height: {measurement_results['height']:.2f}")
    print(f"Intensity: {measurement_results['intensity']:.2f}")
