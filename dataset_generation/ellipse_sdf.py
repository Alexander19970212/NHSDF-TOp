import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

def ellipse_sdf(points, a, b):
    """
    Approximate signed distance function for a set of points with respect to an ellipse.
    Positive outside, negative inside, zero on the curve.
    
    Parameters:
    - points: np.array([[x1, y1], [x2, y2], ...]) - array of query points
    - center: np.array([cx, cy]) - ellipse center
    - a: float - semi-major axis
    - b: float - semi-minor axis
    """
    # Translate points to origin-centered ellipse coordinate system
    translated_points = points
    
    # Scale to unit circle
    x_scaled = translated_points[:, 0] / a
    y_scaled = translated_points[:, 1] / b
    
    # Compute distance to unit circle in scaled space
    scaled_dist = np.sqrt(x_scaled**2 + y_scaled**2)
    
    # Handle case where scaled_dist is zero
    min_axis = min(a, b)
    scaled_dist[scaled_dist == 0] = min_axis
    
    # Approximate the actual distance
    # Inside: negative, Outside: positive
    return scaled_dist * min_axis - min_axis

def generate_ellipse_sdf_dataset(num_ellipse=1000,
                                 points_per_ellipse=500,
                                 smooth_factor=44,
                                 filename='shape_datasets/ellipse_sdf_dataset.csv'):
    """
    Generate a dataset of points and their SDFs for random ellipses.
    Each ellipse is defined by its center, semi-major axis, semi-minor axis and rotation angle.
    """
    data = []
    
    for _ in tqdm(range(num_ellipse)):
        # Generate random ellipse parameters
        center = np.array([0, 0])  # Center fixed at (0, 0)
        # a = np.random.uniform(0.2, 0.8)  # Semi-major axis
        a = 0.5
        b_w = np.random.uniform(0.5, 1.5)  # Semi-minor axis (smaller than a)
        b = a * b_w
        # Generate random points
        points = np.random.uniform(-1, 1, (points_per_ellipse, 2))

        sdf = ellipse_sdf(points, a, b)
        sdf = 1/(1 + np.exp(smooth_factor*sdf))
        
        for i, point in enumerate(points):
            data.append([
                point[0], point[1],  # Point coordinates
                b_w,  # Semi-axes ratio
                sdf[i],
                1
            ])
    
    # Create DataFrame
    columns = [
        'point_x', 'point_y',
        'semi_axes_ratio',
        'sdf',
        'arc_ratio'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    return df

def generate_ellipse_sdf_surface_dataset(
        num_ellipse=1000,
        points_per_ellipse=1000,
        smooth_factor=44,
        filename='../shape_datasets/ellipse_sdf_surface_dataset_test',
        axes_length=1):
    """
    Generate a dataset of points and their SDFs for random ellipses.
    Each ellipse is defined by its center, semi-major axis, semi-minor axis and rotation angle.
    """
    # Lists to store our data
    data = []
    # Create a grid of points
    point_per_side = int(np.sqrt(points_per_ellipse))
    x = np.linspace(-axes_length, axes_length, point_per_side)
    y = np.linspace(-axes_length, axes_length, point_per_side)
    X, Y = np.meshgrid(x, y)
    points = np.array([X.flatten(), Y.flatten()]).T
    
    for _ in tqdm(range(num_ellipse)):
        # Generate random ellipse parameters
        center = np.array([0, 0])  # Center fixed at (0, 0)
        # a = np.random.uniform(0.2, 0.8)  # Semi-major axis
        a = 0.5
        b_w = np.random.uniform(0.5, 1.5)  # Semi-minor axis (smaller than a)
        b = a * b_w
        
        sdf = ellipse_sdf(points, a, b)
        sdf = 1/(1 + np.exp(smooth_factor*sdf))

        sdf_str = ','.join(map(str, sdf.tolist()))
        
        data.append([
            b_w,  # Semi-axes ratio
            sdf_str
        ])
    
    # Create DataFrame
    columns = [
        'semi_axes_ratio',
        'sdf'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(f'{filename}.csv', index=False)
    print(f"Dataset saved to {filename}")

    # Save points grid for later use
    points_df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1]
    })
    points_df.to_csv(f'{filename}_grid.csv', index=False)
    print("Points grid saved to points_grid.csv")
    
    return df, points_df

def plot_ellipse_sdf_dataset(df, points_per_ellipse=1000):
    # Plot a few examples to verify
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        # Get data for one ellipse
        ellipse_data = df.iloc[i*points_per_ellipse:(i+1)*points_per_ellipse]
        b_w = ellipse_data['semi_axes_ratio'].iloc[0]
        a = 0.5
        b = a * b_w

        print(a, b)
        
        # Create ellipse patch
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(np.array([0, 0]), 2*a, 2*b, fill=False, color='black')
        
        
        # Plot points, colored by SDF
        scatter = axes[i].scatter(ellipse_data['point_x'], 
                                ellipse_data['point_y'],
                                c=ellipse_data['sdf'],
                                cmap='RdBu')
        plt.colorbar(scatter, ax=axes[i])
        axes[i].add_patch(ellipse)
        axes[i].set_aspect('equal')
        axes[i].set_title(f'Ellipse {i+1}')

    plt.tight_layout()
    plt.show()