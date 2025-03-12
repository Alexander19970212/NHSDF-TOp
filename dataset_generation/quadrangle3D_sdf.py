# dataset_generation/quadrangle_sdf.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils_generation import point_to_line_distance

from utils_generation import SDF_polygon_3D
from quadrangle_sdf import generate_quadrangle

import os
import argparse

#################################################################################################################

####### dataset generation with rounded corners #############
from utils_generation import get_rounded_polygon_segments_rand_radius, signed_distance_polygon, compute_perimeter

def generate_quadrangle_3DHeavisideSDF(
        num_quadrangle=1000,
        smooth_factor=40,
        min_radius=0.01,
        max_radius_limit=3,
        store_dir='quadrangle_3DHeavisideSDF'
):
    """
    Generate dataset of signed distances for random quadrangle
    
    Parameters:
    - num_quadrangle: number of random quadrangle to generate
    - points_per_quadrangle: number of random points to sample per quadrangle
    - filename: output CSV file name
    """
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    
    # Generate multiple triangles
    for quadrangle_index in tqdm(range(num_quadrangle)):
        # Generate random quadrangle vertices
        while True:
            vertices = generate_quadrangle()
            # vertices = generate_triangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, min_radius, max_radius_limit=max_radius_limit))
            if arcs_intersection == False:
                break

        v1, v2, v3, v4 = vertices
        arc_radii = np.array([radius for _, _, _, radius in arc_segments])
        perimeter, line_perimeter, arc_perimeter = compute_perimeter(line_segments, arc_segments)
        arc_ratio = arc_perimeter / perimeter
        arc_centers = np.array([center for _, _, center, _ in arc_segments])

        z_level = np.random.uniform(-1, 1)

        # Generate random points
        # Sample more points near the triangle
        quadrangle_center = (v1 + v2 + v3 + v4) / 4

        # points on border
        point_per_side = 30  # Adjust this for desired grid density
        x = np.linspace(-1, 1, point_per_side)
        y = np.linspace(-1, 1, point_per_side) 
        z = np.linspace(-1, 1, point_per_side)

        # Create meshgrid
        X, Y, Z = np.meshgrid(x, y, z)

        # Reshape to get array of 3D points
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        noise_scale = 0.01  # Define the noise scale for small perturbations
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=points.shape)
        points = points + noise

        sdf_3d = SDF_polygon_3D(points, z_level, line_segments, arc_segments, vertices)

        sdf_abs = np.abs(sdf_3d)
        filter_bids = np.argsort(sdf_abs)[:1000]

        filtered_points = points[filter_bids]
        filtered_sdf = sdf_3d[filter_bids]

        points_gaussian = np.random.normal(loc=np.append(quadrangle_center, 0.5), scale=0.5, size=(1000, 3))
        points_gaussian = np.clip(points_gaussian, -1, 1)

        sdf_3d_gaussian = SDF_polygon_3D(points_gaussian, z_level, line_segments, arc_segments, vertices)

        points_for_dataset = np.vstack([filtered_points, points_gaussian])
        sdf_3d_for_dataset = np.concatenate([filtered_sdf, sdf_3d_gaussian])

        heaviside_sdf = 1/(1 + np.exp(-smooth_factor*sdf_3d_for_dataset))

        # Lists to store our data
        data = []
        
        # Calculate signed distance for each point
        for i, point in enumerate(points_for_dataset):
            
            row = [
                point[0], point[1], point[2],  # point coordinates
                v3[0], v3[1],        # third vertex
                v4[0], v4[1],        # fourth vertex
                arc_radii[0]/max_radius_limit, # normalized radius
                arc_radii[1]/max_radius_limit, # normalized radius
                arc_radii[2]/max_radius_limit, # normalized radius
                arc_radii[3]/max_radius_limit, # normalized radius
                z_level,
                heaviside_sdf[i],
                arc_ratio
            ]
            data.append(row)
    
        # Convert to DataFrame
        columns = [
            'point_x', 'point_y', 'point_z',
            'v3_x', 'v3_y',
            'v4_x', 'v4_y',
            'r_q1', 'r_q2', 'r_q3', 'r_q4', # q means quadrangle
            'z_level',
            'heaviside_sdf',
            'arc_ratio'
        ]
        df = pd.DataFrame(data, columns=columns)
        
        # Save to CSV
        df.to_csv(f'{store_dir}/{quadrangle_index}.csv', index=False)
        # print(f"Dataset saved to {store_dir}/{quadrangle_index}.csv")
    
    return df

def generate_quadrangle_reconstruction_dataset(
        num_quadrangle=1000,
        smooth_factor=40,
        filename='quadrangle_reconstruction_dataset',
        axes_length=1,
        max_radius_limit=3):
    """
    Generate dataset of signed distances for random quadrangle
    
    Parameters:
    - num_quadrangle: number of random quadrangle to generate
    - filename: output CSV file name
    """
    # Lists to store our data
    data = []
    # Create a grid of points

    # Generate multiple triangles
    for _ in tqdm(range(num_quadrangle)):
        # Generate random quadrangle vertices
        while True:
            vertices = generate_quadrangle()
            # vertices = generate_triangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, 0.1, max_radius_limit=max_radius_limit))
            if arcs_intersection == False:
                break

        v1, v2, v3, v4 = vertices
        arc_radii = np.array([radius for _, _, _, radius in arc_segments])

        perimeter, line_perimeter, arc_perimeter = compute_perimeter(line_segments, arc_segments)
        arc_ratio = arc_perimeter / perimeter

        # Store data as a row: [point_x, point_y, v1_x, v1_y, v2_x, v2_y, v3_x, v3_y, v4_x, v4_y, signed_distance]
        row = [
            v3[0], v3[1],        # third vertex
            v4[0], v4[1],        # fourth vertex
            arc_radii[0]/max_radius_limit, # normalized radius
            arc_radii[1]/max_radius_limit, # normalized radius
            arc_radii[2]/max_radius_limit, # normalized radius
            arc_radii[3]/max_radius_limit, # normalized radius
            arc_ratio
        ]
        data.append(row)
    
    # Convert to DataFrame
    columns = [
        'v3_x', 'v3_y',
        'v4_x', 'v4_y',
        'r_q1', 'r_q2', 'r_q3', 'r_q4', # q means quadrangle
        'arc_ratio'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(f'{filename}.csv', index=False)
    print(f"Dataset saved to {filename}")

    return df

#############################################################

def generate_quadrangle_random_radius_dataset(
        num_quadrangle=100,
        sample_per_quadrangle=100,
        smooth_factor=40,
        filename='quadrangle_random_radius_dataset',
        max_radius_limit=3,
        axes_length=1   
):
    """
    Generate dataset of signed distances for random quadrangle
    
    Parameters:
    - num_quadrangle: number of random quadrangle to generate
    - sample_per_quadrangle: number sample with random radiuses for each quadrangle
    - filename: output CSV file name
    """
    # Lists to store our data
    data = []
    quadrangle_count = 0
    
    # Generate multiple triangles
    from tqdm import tqdm

    with tqdm(total=num_quadrangle, desc="Quadrangle") as pbar:
        while True:
            vertices = generate_quadrangle()

            failed_count = 0
            success_count = 0
            while True:
                line_segments, arc_segments, arcs_intersection = (
                    get_rounded_polygon_segments_rand_radius(vertices, 0.1, max_radius_limit=max_radius_limit))
                
                if arcs_intersection == True:
                    failed_count += 1
                    if failed_count > 10:
                        break
                    continue
                else:
                    failed_count = 0
                    
                    v1, v2, v3, v4 = vertices
                    arc_radii = np.array([radius for _, _, _, radius in arc_segments])

                    perimeter, line_perimeter, arc_perimeter = compute_perimeter(line_segments, arc_segments)
                    arc_ratio = arc_perimeter / perimeter

                    row = [ f'qd_{quadrangle_count}',
                            v3[0], v3[1],        # third vertex
                            v4[0], v4[1],        # fourth vertex
                            arc_radii[0]/max_radius_limit, # normalized radius
                            arc_radii[1]/max_radius_limit, # normalized radius
                            arc_radii[2]/max_radius_limit, # normalized radius
                            arc_radii[3]/max_radius_limit, # normalized radius
                            arc_ratio
                        ]
                    data.append(row)

                    success_count += 1
                    if success_count >= sample_per_quadrangle:
                        quadrangle_count += 1
                        pbar.update(1)
                        break

            if quadrangle_count >= num_quadrangle:
                break

    # Convert to DataFrame
    columns = [
        'feature_id',
        'v3_x', 'v3_y',
        'v4_x', 'v4_y',
        'r_q1', 'r_q2', 'r_q3', 'r_q4', # q means quadrangle
        'arc_ratio'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(f'{filename}.csv', index=False)
    print(f"Dataset saved to {filename}")

    return df

#################################################################################################################

def main(args):
    print(args)
    if args.mode == '3d_heaviside_sdf':
        generate_quadrangle_3DHeavisideSDF(num_quadrangle=args.num_quadrangle,
                                          smooth_factor=args.smooth_factor,
                                          min_radius=args.min_radius,
                                          max_radius_limit=args.max_radius_limit,
                                          store_dir=args.store_dir)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='3d_heaviside_sdf', help='Mode to run')
    parser.add_argument('--num_quadrangle', type=int, default=10000, help='Number of quadrangle to generate')
    parser.add_argument('--smooth_factor', type=int, default=20, help='Smooth factor')
    parser.add_argument('--min_radius', type=float, default=0.01, help='Minimum radius')
    parser.add_argument('--max_radius_limit', type=float, default=3, help='Maximum radius limit')
    parser.add_argument('--store_dir', type=str, default='quadrangle_3DHeavisideSDF', help='Store directory')
    args = parser.parse_args()