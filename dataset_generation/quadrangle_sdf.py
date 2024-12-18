# dataset_generation/quadrangle_sdf.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils_generation import point_to_line_distance

def generate_quadrangle():
    """
    Generate a random quadrangle without self-intersections in [0, 1] domain.
    Returns vertices as a numpy array of shape (4, 2).
    """
    while True:
    
        # Calculate vertices
        v1 = np.array([-0.5, -0.5])
        v2 = np.array([0.5, -0.5])
        
        x3 = np.random.uniform(-0.8, 0.8)
        y3 = np.random.uniform(-0.2, 0.8)
        v3 = np.array([x3, y3])

        x4 = np.random.uniform(-0.8, 0.8)
        y4 = np.random.uniform(0.2, 0.8)
        v4 = np.array([x4, y4])

        vertices = np.array([v1, v2, v3, v4])
        
        # Check for self-intersections
        has_intersection = False
        
        # Check each pair of non-adjacent edges
        def segments_intersect(p1, p2, p3, p4):
            """Check if line segments (p1,p2) and (p3,p4) intersect"""
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
            return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
        
        # Check intersection between edges (0,1) and (2,3)
        if segments_intersect(vertices[0], vertices[1], 
                            vertices[2], vertices[3]):
            has_intersection = True
            
        # Check intersection between edges (1,2) and (3,0)
        if segments_intersect(vertices[1], vertices[2], 
                            vertices[3], vertices[0]):
            has_intersection = True

        # Calculate area of the quadrangle by splitting into two triangles
        v1, v2, v3, v4 = vertices
        # Area of triangle v1,v2,v3
        area1 = abs(np.cross(v2-v1, v3-v1)) / 2
        # Area of triangle v1,v3,v4 
        area2 = abs(np.cross(v3-v1, v4-v1)) / 2
        total_area = area1 + area2
        
        # Check if area is too small
        if total_area < 0.35:
            has_intersection = True
        
        if not has_intersection:
            return vertices
        
def point_in_quadrangle(point, vertices):
    """
    Check if point is inside quadrangle using winding number algorithm.
    
    Parameters:
    point: np.array([x, y]) - the point to check
    vertices: np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]]) - quadrangle vertices in CCW order
    
    Returns:
    bool: True if point is inside, False otherwise
    """
    def is_left(p0, p1, p2):
        """
        Test if point p2 is left/on/right of line from p0 to p1.
        Returns:
        > 0 for p2 left of the line
        = 0 for p2 on the line
        < 0 for p2 right of the line
        """
        return ((p1[0] - p0[0]) * (p2[1] - p0[1]) - 
                (p2[0] - p0[0]) * (p1[1] - p0[1]))

    wn = 0  # winding number counter

    # Loop through all edges of the polygon
    for i in range(len(vertices)):
        # Get current and next vertex
        current = vertices[i]
        next = vertices[(i + 1) % len(vertices)]

        # Test if a point is left/on/right of an edge
        if current[1] <= point[1]:  # start y <= point y
            if next[1] > point[1]:  # an upward crossing
                if is_left(current, next, point) > 0:  # point left of edge
                    wn += 1  # valid up intersect
        else:  # start y > point y
            if next[1] <= point[1]:  # a downward crossing
                if is_left(current, next, point) < 0:  # point right of edge
                    wn -= 1  # valid down intersect

    return wn != 0

def signed_distance_quadrangle(point, v1, v2, v3, v4, smooth_factor=40, corner_radius=0.1):
    """Calculate signed distance from point to quadrangle"""
    d1 = point_to_line_distance(point, v1, v2)
    d2 = point_to_line_distance(point, v2, v3)
    d3 = point_to_line_distance(point, v3, v4)
    d4 = point_to_line_distance(point, v4, v1)
    
    min_dist = min(d1, d2, d3, d4)
    is_inside = point_in_quadrangle(point, [v1, v2, v3, v4])
    
    if is_inside:
        dist = min_dist
    else:
        dist = -min_dist

    return 1/(1 + np.exp(-smooth_factor*(dist + corner_radius)))

def generate_quadrangle_sdf_dataset(num_quadrangle=1000,
                                    points_per_quadrangle=100,
                                    smooth_factor=40,
                                    filename='quadrangle_sdf_dataset.csv'):
    """
    Generate dataset of signed distances for random quadrangle
    
    Parameters:
    - num_quadrangle: number of random quadrangle to generate
    - points_per_quadrangle: number of random points to sample per quadrangle
    - filename: output CSV file name
    """
    # Lists to store our data
    data = []
    
    # Generate multiple triangles
    for _ in tqdm(range(num_quadrangle)):
        # Generate random quadrangle vertices
        quadrangle = generate_quadrangle()
        v1, v2, v3, v4 = quadrangle
        
        # Generate random points
        # Sample more points near the triangle
        quadrangle_center = (v1 + v2 + v3 + v4) / 4
        
        # Mix of uniform and gaussian sampling
        num_uniform = points_per_quadrangle // 2
        num_gaussian = points_per_quadrangle - num_uniform
        
        # Uniform sampling in the bounding box
        points_uniform = np.random.rand(num_uniform, 2)*2 - 1
        
        # Gaussian sampling around the triangle
        points_gaussian = np.random.normal(loc=quadrangle_center, scale=0.5, size=(num_gaussian, 2))
        points_gaussian = np.clip(points_gaussian, -1, 1)
        
        # Combine points
        points = np.vstack([points_uniform, points_gaussian])
        
        # Calculate signed distance for each point
        for point in points:
            sdf = signed_distance_quadrangle(point, v1, v2, v3, v4, smooth_factor=smooth_factor)
            
            # Store data as a row: [point_x, point_y, v1_x, v1_y, v2_x, v2_y, v3_x, v3_y, v4_x, v4_y, signed_distance]
            row = [
                point[0], point[1],  # point coordinates
                v3[0], v3[1],        # third vertex
                v4[0], v4[1],        # fourth vertex
                sdf                  # signed distance value
            ]
            data.append(row)
    
    # Convert to DataFrame
    columns = [
        'point_x', 'point_y',
        'v3_x', 'v3_y',
        'v4_x', 'v4_y',
        'sdf'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    return df

#################################################################################################################

####### dataset generation with rounded corners #############
from utils_generation import get_rounded_polygon_segments_rand_radius, signed_distance_polygon

def generate_rounded_quadrangle_sdf_dataset(
        num_quadrangle=1000,
        points_per_quadrangle=100,
        smooth_factor=40,
        filename='quadrangle_sdf_dataset.csv'
):
    """
    Generate dataset of signed distances for random quadrangle
    
    Parameters:
    - num_quadrangle: number of random quadrangle to generate
    - points_per_quadrangle: number of random points to sample per quadrangle
    - filename: output CSV file name
    """
    # Lists to store our data
    data = []
    
    # Generate multiple triangles
    for _ in tqdm(range(num_quadrangle)):
        # Generate random quadrangle vertices
        while True:
            vertices = generate_quadrangle()
            # vertices = generate_triangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, 0.1))
            if arcs_intersection == False:
                break

        v1, v2, v3, v4 = vertices
        arc_radii = np.array([radius for _, _, _, radius in arc_segments])
        
        # Generate random points
        # Sample more points near the triangle
        quadrangle_center = (v1 + v2 + v3 + v4) / 4
        
        # Mix of uniform and gaussian sampling
        num_uniform = points_per_quadrangle // 2
        num_gaussian = points_per_quadrangle - num_uniform
        
        # Uniform sampling in the bounding box
        points_uniform = np.random.rand(num_uniform, 2)*2 - 1
        
        # Gaussian sampling around the triangle
        points_gaussian = np.random.normal(loc=quadrangle_center, scale=0.5, size=(num_gaussian, 2))
        points_gaussian = np.clip(points_gaussian, -1, 1)
        
        # Combine points
        points = np.vstack([points_uniform, points_gaussian])

        sdf = signed_distance_polygon(points, line_segments, arc_segments, vertices, smooth_factor=smooth_factor)
        
        # Calculate signed distance for each point
        for i, point in enumerate(points):
            
            row = [
                point[0], point[1],  # point coordinates
                v3[0], v3[1],        # third vertex
                v4[0], v4[1],        # fourth vertex
                arc_radii[0],
                arc_radii[1],
                arc_radii[2],
                arc_radii[3],
                sdf[i]                  # signed distance value
            ]
            data.append(row)
    
    # Convert to DataFrame
    columns = [
        'point_x', 'point_y',
        'v3_x', 'v3_y',
        'v4_x', 'v4_y',
        'r_q1', 'r_q2', 'r_q3', 'r_q4', # q means quadrangle
        'sdf'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    return df

def generate_rounded_quadrangle_sdf_surface_dataset(
        num_quadrangle=1000,
        points_per_quadrangle=1000,
        smooth_factor=40,
        filename='../shape_datasets/quadrangle_sdf_surface_dataset_test',
        axes_length=1):
    """
    Generate dataset of signed distances for random quadrangle
    
    Parameters:
    - num_quadrangle: number of random quadrangle to generate
    - points_per_quadrangle: number of random points to sample per quadrangle
    - filename: output CSV file name
    """
    # Lists to store our data
    data = []
    # Create a grid of points
    point_per_side = int(np.sqrt(points_per_quadrangle))
    x = np.linspace(-axes_length, axes_length, point_per_side)
    y = np.linspace(-axes_length, axes_length, point_per_side)
    X, Y = np.meshgrid(x, y)
    points = np.array([X.flatten(), Y.flatten()]).T

    # Generate multiple triangles
    for _ in tqdm(range(num_quadrangle)):
        # Generate random quadrangle vertices
        while True:
            vertices = generate_quadrangle()
            # vertices = generate_triangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, 0.1))
            if arcs_intersection == False:
                break

        v1, v2, v3, v4 = vertices
        arc_radii = np.array([radius for _, _, _, radius in arc_segments])

        sdf = signed_distance_polygon(points, line_segments, arc_segments, vertices, smooth_factor=smooth_factor)

        # print(sdf)
        
        # Convert sdf list to a string representation
        sdf_str = ','.join(map(str, sdf.tolist()))
        # print(sdf_str)
        
        # Store data as a row: [point_x, point_y, v1_x, v1_y, v2_x, v2_y, v3_x, v3_y, v4_x, v4_y, signed_distance]
        row = [
            v3[0], v3[1],        # third vertex
            v4[0], v4[1],        # fourth vertex
            arc_radii[0],
            arc_radii[1],
            arc_radii[2],
            arc_radii[3],
            sdf_str               # signed distance value as string
        ]
        data.append(row)
    
    # Convert to DataFrame
    columns = [
        'v3_x', 'v3_y',
        'v4_x', 'v4_y',
        'r_q1', 'r_q2', 'r_q3', 'r_q4', # q means quadrangle
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

#################################################################################################################

# testing the function
def plot_random_quadrangles():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    
    for i in range(4):
        vertices = generate_quadrangle()
        vertices_plot = np.vstack([vertices, vertices[0]])
        
        axes[i].plot(vertices_plot[:, 0], vertices_plot[:, 1], 'b-', linewidth=2)
        axes[i].fill(vertices[:, 0], vertices[:, 1], alpha=0.3)
        axes[i].scatter(vertices[:, 0], vertices[:, 1], c='red', s=100)
        
        # Add vertex labels
        for j, (x, y) in enumerate(vertices):
            axes[i].annotate(f'v{j}', (x, y), xytext=(5, 5), 
                           textcoords='offset points')
        
        axes[i].grid(True)
        axes[i].set_xlim(-1.1, 1.1)
        axes[i].set_ylim(-1.1, 1.1)
        axes[i].set_aspect('equal')
        axes[i].set_title(f'Quadrangle {i+1}')
    
    plt.tight_layout()
    plt.show()

def plot_sdf_random_quadrangle(smooth_factor=40):
    # Generate random quadrangle vertices
    quadrangle = generate_quadrangle()
    v1, v2, v3, v4 = quadrangle

    # Create a grid of points
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x, y)

    # Calculate signed distance for each point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i,j], Y[i,j]])
            Z[i,j] =  signed_distance_quadrangle(point, v1, v2, v3, v4, smooth_factor=smooth_factor)

    # Create plots
    fig = plt.figure(figsize=(20, 5))

    # 2D contour plot
    ax1 = fig.add_subplot(131)
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='RdBu')
    ax1.plot([v1[0], v2[0], v3[0], v4[0], v1[0]], [v1[1], v2[1], v3[1], v4[1], v1[1]], 'k-', linewidth=2)
    plt.colorbar(contour, ax=ax1)
    ax1.set_title('2D Contour Plot of Signed Distance')
    ax1.set_aspect('equal')

    # 3D surface plot
    ax2 = fig.add_subplot(132, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='RdBu')
    plt.colorbar(surf, ax=ax2)
    ax2.set_title('3D Surface Plot of Signed Distance')

    # 2D contour plot with more levels
    ax3 = fig.add_subplot(133)
    contour = ax3.contour(X, Y, Z, levels=50, cmap='RdBu')
    ax3.plot([v1[0], v2[0], v3[0], v4[0], v1[0]], [v1[1], v2[1], v3[1], v4[1], v1[1]], 'k-', linewidth=2)
    plt.colorbar(contour, ax=ax3)
    ax3.set_title('2D Contour Lines')
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def plot_quadrangle_sdf_dataset(df, points_per_quadrangle=500):
    # Plot first few triangles and their points
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        # Get data for one quadrangle
        quadrangle_data = df.iloc[i*points_per_quadrangle:(i+1)*points_per_quadrangle]
        v1 = np.array([-0.5, -0.5])
        v2 = np.array([0.5, -0.5])
        v3 = np.array([quadrangle_data['v3_x'].iloc[0], quadrangle_data['v3_y'].iloc[0]])
        v4 = np.array([quadrangle_data['v4_x'].iloc[0], quadrangle_data['v4_y'].iloc[0]])
        
        # Plot quadrangle
        axes[i].plot([v1[0], v2[0], v3[0], v4[0], v1[0]], 
                    [v1[1], v2[1], v3[1], v4[1], v1[1]], 'k-')
        
        # Plot points, colored by SDF
        scatter = axes[i].scatter(quadrangle_data['point_x'], 
                                quadrangle_data['point_y'],
                                c=quadrangle_data['sdf'],
                                cmap='RdBu')
        plt.colorbar(scatter, ax=axes[i])
        axes[i].set_aspect('equal')
        axes[i].set_title(f'Triangle {i+1}')

    plt.tight_layout()
    plt.show()