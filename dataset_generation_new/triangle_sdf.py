import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_generation import point_to_line_distance
import matplotlib.pyplot as plt
import os

def point_in_triangle(point, v1, v2, v3):
        """Check if point is inside triangle using barycentric coordinates"""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(point, v1, v2)
        d2 = sign(point, v2, v3)
        d3 = sign(point, v3, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

def triangle_center(v1, v2, v3):
    """Calculate centroid of triangle"""
    return (v1 + v2 + v3) / 3.0

def signed_distance(point, v1, v2, v3, smooth_factor=20,     corner_radius=0.1):
    """Calculate signed distance from point to triangle with rounded inner corners"""
    d1 = point_to_line_distance(point, v1, v2)
    d2 = point_to_line_distance(point, v2, v3)
    d3 = point_to_line_distance(point, v3, v1)
    
    min_dist = min(d1, d2, d3)
    is_inside = point_in_triangle(point, v1, v2, v3)
    
    if is_inside:
        dist = min_dist
    else:
        dist = -min_dist

    return 1/(1 + np.exp(-smooth_factor*(dist + corner_radius)))

def generate_triangle():
    while True:
        # Generate vertices for the triangle
        v1 = np.array([-0.5, -0.5])  # First point of the edge parallel to x-axis
        v2 = np.array([0.5, -0.5])   # Second point of the edge parallel to x-axis
        
        # Generate the third vertex randomly
        x3 = np.random.uniform(-0.8, 0.8)
        y3 = np.random.uniform(0.1, 0.8)
        v3 = np.array([x3, y3])
        
        # No need for rotation or translation as the triangle is already in the desired position
        
        # Check if the triangle meets our criteria (all vertices within the unit circle)
        if np.all(np.linalg.norm([v1, v2, v3], axis=1) <= 1):
            break
        # Calculate triangle area using cross product
        area = abs(np.cross(v2-v1, v3-v1)) / 2
        
        # Only accept triangles with area > 0.1 (arbitrary threshold)
        if area > 0.2:
            break

    vertices = np.array([v1, v2, v3])
    return vertices

def generate_triangle_sdf_dataset(num_triangle=100, points_per_triangle=1000, smooth_factor=40, filename='triangle_sdf_dataset_short.csv'):
    """
    Generate dataset of signed distances for random triangles
    
    Parameters:
    - num_triangles: number of random triangles to generate
    - points_per_triangle: number of random points to sample per triangle
    - filename: output CSV file name
    """
    # Lists to store our data
    data = []
    
    # Generate multiple triangles
    for _ in tqdm(range(num_triangle)):
        # Keep generating triangles until we get one with sufficient area
        vertices = generate_triangle()
        v1, v2, v3 = vertices[0], vertices[1], vertices[2]
        
        # Generate random points
        # Sample more points near the triangle
        triangle_center = (v1 + v2 + v3) / 3
        
        # Mix of uniform and gaussian sampling
        num_uniform = points_per_triangle // 2
        num_gaussian = points_per_triangle - num_uniform
        
        # Uniform sampling in the bounding box
        points_uniform = np.random.rand(num_uniform, 2)*2 - 1
        
        # Gaussian sampling around the triangle
        points_gaussian = np.random.normal(loc=triangle_center, scale=0.5, size=(num_gaussian, 2))
        points_gaussian = np.clip(points_gaussian, -1, 1)
        
        # Combine points
        points = np.vstack([points_uniform, points_gaussian])
        
        # Calculate signed distance for each point
        for point in points:
            sdf = signed_distance(point, v1, v2, v3, smooth_factor=smooth_factor)
            
            # Store data as a row: [point_x, point_y, v1_x, v1_y, v2_x, v2_y, v3_x, v3_y, signed_distance]
            row = [
                point[0], point[1],  # point coordinates
                v3[0], v3[1],        # third vertex
                sdf                  # signed distance value
            ]
            data.append(row)
    
    # Convert to DataFrame
    columns = [
        'point_x', 'point_y',
        'v1_x', 'v1_y',
        'sdf'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    return df

#################################################################################################################
from utils_generation import get_rounded_polygon_segments_rand_radius, signed_distance_polygon, compute_perimeter
from utils_generation import contour_points_generator

def generate_rounded_triangle_sdf_dataset(
        num_triangle=1000,
        points_per_triangle=100,
        smooth_factor=40,
        min_radius=0.01,
        max_radius_limit=3,
        filename='triangle_sdf_dataset.csv'
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
    for _ in tqdm(range(num_triangle)):
        # Generate random quadrangle vertices
        while True:
            vertices = generate_triangle()
            # vertices = generate_triangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, min_radius, max_radius_limit=max_radius_limit))
            if arcs_intersection == False:
                break

        ###########################################
        perimeter, line_perimeter, arc_perimeter = compute_perimeter(line_segments, arc_segments)
        arc_ratio = arc_perimeter / perimeter

        v1, v2, v3 = vertices
        arc_radii = np.array([radius for _, _, _, radius in arc_segments])
        arc_centers = np.array([center for _, _, center, _ in arc_segments])
        ##############################################
        # # Generate random points
        # # Sample more points near the triangle
        # triangle_center = (v1 + v2 + v3) / 3
        
        # # Mix of uniform and gaussian sampling
        # # num_uniform = points_per_triangle // 3
        # num_gaussian = points_per_triangle // 3
        # num_points_in_vertices = points_per_triangle // 3
        # # num_points_in_vertices = points_per_triangle

        # # Uniform sampling in the bounding box
        # # points_uniform = np.random.rand(num_uniform, 2)*2 - 1
        
        # # Gaussian sampling around the triangle
        # points_gaussian = np.random.normal(loc=triangle_center, scale=0.5, size=(num_gaussian, 2))
        # points_gaussian = np.clip(points_gaussian, -1, 1)

        # # Generate points in the vertices
        # points_in_vertices = []
        # for center, radius in zip([v1, v2, v3], arc_radii):
        #     points_in_circle = np.random.normal(loc=center, scale=0.2, size=(num_points_in_vertices // 3, 2))
        #     points_in_circle = np.clip(points_in_circle, -1, 1)
        #     points_in_vertices.append(points_in_circle)
        
        # points_in_vertices = np.vstack(points_in_vertices)

        # num_uniform = points_per_triangle - points_in_vertices.shape[0] - points_gaussian.shape[0]
        # points_uniform = np.random.rand(num_uniform, 2)*2 - 1
        
        # # Combine points
        # points = np.vstack([points_uniform, points_gaussian, points_in_vertices])
        # # points = points_in_vertices

        # sdf = signed_distance_polygon(points, line_segments, arc_segments, vertices, smooth_factor=smooth_factor)

        ###########################################

        points, sdf = contour_points_generator(signed_distance_polygon, (line_segments, arc_segments, vertices, smooth_factor),
                                               points_per_triangle)
        
        # Calculate signed distance for each point
        for i, point in enumerate(points):
            row = [
                point[0], point[1],  # point coordinates
                v3[0], v3[1],        # third vertex
                arc_radii[0]/max_radius_limit, # normalized radius
                arc_radii[1]/max_radius_limit, # normalized radius
                arc_radii[2]/max_radius_limit, # normalized radius
                sdf[i],
                arc_ratio
            ]
            data.append(row)
    
    # Convert to DataFrame
    columns = [
        'point_x', 'point_y',
        'v1_x', 'v1_y',
        'r_t1', 'r_t2', 'r_t3', # t means triangle
        'sdf',
        'arc_ratio'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    return df

# def generate_rounded_triangle_sdf_surface_dataset(
#         num_triangle=1000,
#         points_per_triangle=100,
#         smooth_factor=40,
#         filename='rounded_triangle_sdf_surface_dataset.csv'
# ):
#     """
#     Generate dataset of signed distances for random quadrangle
    
#     Parameters:
#     - num_quadrangle: number of random quadrangle to generate
#     - points_per_quadrangle: number of random points to sample per quadrangle
#     - filename: output CSV file name
#     """
#     # Lists to store our data
#     data = []
#     # Create a grid of points
#     point_per_side = int(np.sqrt(points_per_triangle))
#     x = np.linspace(-1, 1, point_per_side)
#     y = np.linspace(-1, 1, point_per_side)
#     X, Y = np.meshgrid(x, y)
#     points = np.array([X.flatten(), Y.flatten()]).T
    
#     # Generate multiple triangles
#     for _ in tqdm(range(num_triangle)):
#         # Generate random quadrangle vertices
#         while True:
#             vertices = generate_triangle()
#             # vertices = generate_triangle()
#             line_segments, arc_segments, arcs_intersection = (
#                 get_rounded_polygon_segments_rand_radius(vertices, 0.1))
#             if arcs_intersection == False:
#                 break

#         v1, v2, v3 = vertices
#         arc_radii = np.array([radius for _, _, _, radius in arc_segments])

#         sdf = signed_distance_polygon(points, line_segments, arc_segments, vertices, smooth_factor=smooth_factor)
        
#         # Calculate signed distance for each point
        
#         row = [
#                 v3[0], v3[1],        # third vertex
#                 arc_radii[0],
#                 arc_radii[1],
#                 arc_radii[2],
#                 sdf                  # signed distance value
#         ]
#         data.append(row)
    
#     # Convert to DataFrame
#     columns = [
#         'v1_x', 'v1_y',
#         'r_t1', 'r_t2', 'r_t3', # t means triangle
#         'sdf'
#     ]
    
#     df = pd.DataFrame(data, columns=columns)
    
#     # Save to CSV
#     df.to_csv(filename, index=False)
#     print(f"Dataset saved to {filename}")

#     # Save points grid for later use
#     points_df = pd.DataFrame({
#         'x': points[:, 0],
#         'y': points[:, 1]
#     })
#     points_df.to_csv(f'{filename}_grid.csv', index=False)
#     print("Points grid saved to points_grid.csv")
    
#     return df


def generate_rounded_triangle_sdf_surface_dataset(
        num_triangle=1000,
        points_per_triangle=100,
        smooth_factor=40,
        filename='rounded_triangle_sdf_surface_dataset',
        axes_length=1,
        max_radius_limit=3,
        min_radius=0.01
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
    # Create a grid of points
    point_per_side = int(np.sqrt(points_per_triangle))
    x = np.linspace(-axes_length, axes_length, point_per_side)
    y = np.linspace(-axes_length, axes_length, point_per_side)
    X, Y = np.meshgrid(x, y)
    points = np.array([X.flatten(), Y.flatten()]).T

    
    # Generate multiple triangles
    for _ in tqdm(range(num_triangle)):
        # Generate random quadrangle vertices
        while True:
            vertices = generate_triangle()
            # vertices = generate_triangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, min_radius, max_radius_limit=max_radius_limit))
            if arcs_intersection == False:
                break

        v1, v2, v3 = vertices
        arc_radii = np.array([radius for _, _, _, radius in arc_segments])

        sdf = signed_distance_polygon(points, line_segments, arc_segments, vertices, smooth_factor=smooth_factor)
        sdf_str = ','.join(map(str, sdf.tolist()))
        # Calculate signed distance for each point
        
        row = [
                v3[0], v3[1],        # third vertex
                arc_radii[0]/max_radius_limit, # normalized radius
                arc_radii[1]/max_radius_limit, # normalized radius
                arc_radii[2]/max_radius_limit, # normalized radius
                sdf_str                  # signed distance value
        ]
        data.append(row)
    
    # Convert to DataFrame
    columns = [
        'v1_x', 'v1_y',
        'r_t1', 'r_t2', 'r_t3', # t means triangle
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

def generate_traingle_random_radius_dataset(
        num_triangle=100,
        sample_per_triangle=100,
        smooth_factor=40,
        filename='triangle_random_radius_dataset',
        axes_length=1,
        max_radius_limit=3,
        min_radius=0.01
):
    """
    Generate dataset of signed distances for random quadrangle
    
    Parameters:
    - num_triangle: number of random triangle to generate
    - sample_per_triangle: number sample with random radiuses for each triangle
    - filename: output CSV file name
    """
    # Lists to store our data
    data = []
    triangle_count = 0
    
    # Generate multiple triangles
    from tqdm import tqdm

    with tqdm(total=num_triangle, desc="Triangles") as pbar:
        while True:
            vertices = generate_triangle()

            failed_count = 0
            success_count = 0
            while True:
                line_segments, arc_segments, arcs_intersection = (
                    get_rounded_polygon_segments_rand_radius(vertices, min_radius, max_radius_limit=max_radius_limit))
                
                if arcs_intersection == True:
                    failed_count += 1
                    if failed_count > 10:
                        break
                    continue
                else:
                    failed_count = 0
                    
                    v1, v2, v3 = vertices
                    arc_radii = np.array([radius for _, _, _, radius in arc_segments])

                    perimeter, line_perimeter, arc_perimeter = compute_perimeter(line_segments, arc_segments)
                    arc_ratio = arc_perimeter / perimeter

                    row = [ f'tr_{triangle_count}',
                            v3[0], v3[1],        # third vertex
                            arc_radii[0]/max_radius_limit, # normalized radius
                            arc_radii[1]/max_radius_limit, # normalized radius
                            arc_radii[2]/max_radius_limit, # normalized radius
                            arc_ratio
                    ]
                    data.append(row)

                    success_count += 1
                    if success_count >= sample_per_triangle:
                        triangle_count += 1
                        pbar.update(1)
                        break

            if triangle_count >= num_triangle:
                break

    # Convert to DataFrame
    columns = [
        'feature_id',
        'v1_x', 'v1_y',
        'r_t1', 'r_t2', 'r_t3', # t means triangle
        'arc_ratio'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(f'{filename}.csv', index=False)
    print(f"Dataset saved to {filename}")

    return df

def generate_traingle_reconstruction_dataset(
        num_triangle=100,
        smooth_factor=40,
        filename='triangle_reconstruction_dataset',
        axes_length=1,
        max_radius_limit=3,
        min_radius=0.01
):
    """
    Generate dataset of signed distances for random quadrangle
    
    Parameters:
    - num_triangle: number of random triangle to generate
    - sample_per_triangle: number sample with random radiuses for each triangle
    - filename: output CSV file name
    """
    # Lists to store our data
    data = []
    triangle_count = 0
    
    # Generate multiple triangles
    from tqdm import tqdm

    # Generate multiple triangles
    for _ in tqdm(range(num_triangle)):
        # Generate random quadrangle vertices
        while True:
            vertices = generate_triangle()
            # vertices = generate_triangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, max_radius_limit=max_radius_limit, min_radius=min_radius))
            if arcs_intersection == False:
                break

        v1, v2, v3 = vertices
        arc_radii = np.array([radius for _, _, _, radius in arc_segments])

        perimeter, line_perimeter, arc_perimeter = compute_perimeter(line_segments, arc_segments)
        arc_ratio = arc_perimeter / perimeter

        row = [
                v3[0], v3[1],        # third vertex
                arc_radii[0]/max_radius_limit, # normalized radius
                arc_radii[1]/max_radius_limit, # normalized radius
                arc_radii[2]/max_radius_limit, # normalized radius
                arc_ratio
        ]
        data.append(row)


    # Convert to DataFrame
    columns = [
        'v1_x', 'v1_y',
        'r_t1', 'r_t2', 'r_t3', # t means triangle
        'arc_ratio'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(f'{filename}.csv', index=False)
    print(f"Dataset saved to {filename}")

    return df

def generate_triangle_vae_dataset(num_triangle=1000,
                                 smooth_factor=10,
                                 dataset_dir=None,
                                 num_golden_triangle=0,
                                 image_size=64,
                                 max_radius_limit=3,
                                 min_radius=0.01):
    """
    Generate a dataset of points and their SDFs for random triangle.
    """
    from utils_generation import chi_indexing

    if dataset_dir is None:
        raise ValueError("dataset_dir must be provided")
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create a grid of points
    point_per_side = image_size
    x = np.linspace(-1, 1, point_per_side)
    y = np.linspace(-1, 1, point_per_side)
    X, Y = np.meshgrid(x, y)
    points = np.array([X.flatten(), Y.flatten()]).T

    chi_size = len(chi_indexing.keys())
    chi = np.zeros(chi_size)
    chi[0] = 0.5

    
    for t_idx in tqdm(range(num_triangle)):
        # Generate random quadrangle parameters
        while True:
            vertices = generate_triangle()
            # vertices = generate_triangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, max_radius_limit=max_radius_limit, min_radius=min_radius))
            if arcs_intersection == False:
                break

        v1, v2, v3 = vertices
        arc_radii = np.array([radius for _, _, _, radius in arc_segments])

        hv_sdf = signed_distance_polygon(points, line_segments, arc_segments, vertices, smooth_factor=smooth_factor)
        
        chi[chi_indexing["x3"]] = v3[0]
        chi[chi_indexing["y3"]] = v3[1]
        
        chi[chi_indexing["R1"]] = arc_radii[0]/max_radius_limit
        chi[chi_indexing["R2"]] = arc_radii[1]/max_radius_limit
        chi[chi_indexing["R3"]] = arc_radii[2]/max_radius_limit

        hv_sdf_2d = hv_sdf.reshape(image_size, image_size)
        filename = f'{dataset_dir}/triangle_{t_idx}.npy'
        data_to_save = {
            "hv_sdf_2d": hv_sdf_2d,
            "chi": chi,
            "golden": False
        }
        np.save(filename, data_to_save)

#################################################################################################################

def plot_triangle_sdf_dataset(df, points_per_triangle=500):
# Plot first few triangles and their points
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        # Get data for one triangle
        triangle_data = df.iloc[i*points_per_triangle:(i+1)*points_per_triangle]
        # v1 = np.array([triangle_data['v1_x'].iloc[0], triangle_data['v1_y'].iloc[0]])
        # v2 = np.array([triangle_data['v2_x'].iloc[0], triangle_data['v2_y'].iloc[0]])
        v1 = np.array([-0.5, -0.5])
        v2 = np.array([0.5, -0.5])
        v3 = np.array([triangle_data['v1_x'].iloc[0], triangle_data['v1_y'].iloc[0]])
        
        # Plot triangle
        axes[i].plot([v1[0], v2[0], v3[0], v1[0]], 
                    [v1[1], v2[1], v3[1], v1[1]], 'k-')
        
        # Plot points, colored by SDF
        scatter = axes[i].scatter(triangle_data['point_x'], 
                                triangle_data['point_y'],
                                c=triangle_data['sdf'],
                                cmap='RdBu')
        plt.colorbar(scatter, ax=axes[i])
        axes[i].set_aspect('equal')
        axes[i].set_title(f'Triangle {i+1}')

    plt.tight_layout()
    plt.show()



    
    