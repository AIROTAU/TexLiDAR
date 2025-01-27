import numpy as np
import plotly.graph_objects as go


# Function to load a .bin Lidar point cloud file
def load_lidar_point_cloud(file_path):
    # Read binary file containing the Lidar data
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points


# Function to create a 3D scatter plot with hover functionality
def visualize_point_cloud(pc):
    # Create a 3D scatter plot using Plotly
    fig = go.Figure(data=go.Scatter3d(
        x=pc[:, 0],  # X coordinates
        y=pc[:, 1],  # Y coordinates
        z=pc[:, 2],  # Z coordinates
        mode='markers',
        marker=dict(size=2, color=pc[:, 2], colorscale='Viridis', opacity=0.8),
        text=['x: {:.2f}, y: {:.2f}, z: {:.2f}'.format(x, y, z) for x, y, z in zip(pc[:, 0], pc[:, 1], pc[:, 2])],
        # Hover text
        hoverinfo='text'  # Show hover text
    ))

    # Define layout for the 3D plot with a black background
    fig.update_layout(
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Z (meters)",
            xaxis=dict(range=[-100, 100], backgroundcolor='rgb(0, 0, 0)', gridcolor='rgb(127, 127, 127)', zerolinecolor='rgb(127, 127, 127)'),
            yaxis=dict(range=[-100, 100], backgroundcolor='rgb(0, 0, 0)', gridcolor='rgb(127, 127, 127)', zerolinecolor='rgb(127, 127, 127)'),
            zaxis=dict(range=[-100, 100], backgroundcolor='rgb(0, 0, 0)', gridcolor='rgb(127, 127, 127)', zerolinecolor='rgb(127, 127, 127)'),
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        paper_bgcolor='rgb(0, 0, 0)',  # Black background for the plot
        plot_bgcolor='rgb(0, 0, 0)'    # Black background for the plot
    )

    # Show the plot
    fig.show()


if __name__ == "__main__":
    # Path to your .bin file
    FILE_PATH = r"C:\temp\I\8344.bin"  # Use raw string to handle the backslashes

    # Load the point cloud data
    points = load_lidar_point_cloud(FILE_PATH)

    # Visualize the point cloud
    visualize_point_cloud(points)
