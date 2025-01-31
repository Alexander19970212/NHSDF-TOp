# examples with generation dataset for ellipsoides
from ellipse_sdf import generate_ellipse_sdf_dataset, plot_ellipse_sdf_dataset
from triangle_sdf import generate_rounded_triangle_sdf_dataset, plot_triangle_sdf_dataset
from quadrangle_sdf import generate_rounded_quadrangle_sdf_dataset, plot_quadrangle_sdf_dataset
from ellipse_sdf import generate_ellipse_sdf_surface_dataset, plot_ellipse_sdf_dataset
from quadrangle_sdf import generate_rounded_quadrangle_sdf_surface_dataset, plot_quadrangle_sdf_dataset
from triangle_sdf import generate_rounded_triangle_sdf_surface_dataset, plot_triangle_sdf_dataset

import argparse

def main(args):
    ellipse_dataset_path = (
        '../../mnt/local/data/kalexu97/topOpt/ellipse_sdf_dataset_smf22_arc_ratio.csv'
    )
    df = generate_ellipse_sdf_dataset(
        num_ellipse=args.num_shapes,
        points_per_ellipse=args.points_per_shape,
        smooth_factor=int(args.smooth_factor*1.1),
        filename=ellipse_dataset_path
    )

    triangle_dataset_path = (
        '../../mnt/local/data/kalexu97/topOpt/rounded_triangle_sdf_dataset_smf20_arc_ratio.csv'
    )
    df = generate_rounded_triangle_sdf_dataset(
        num_triangle=args.num_shapes,
        points_per_triangle=args.points_per_shape,
        smooth_factor=args.smooth_factor,
        filename=triangle_dataset_path
    )

    quadrangle_dataset_path = (
        '../../mnt/local/data/kalexu97/topOpt/rounded_quadrangle_sdf_dataset_smf20_arc_ratio.csv'
    )
    df = generate_rounded_quadrangle_sdf_dataset(
        num_quadrangle=args.num_shapes,
        points_per_quadrangle=args.points_per_shape,
        smooth_factor=args.smooth_factor,
        filename=quadrangle_dataset_path
    )

    ellipse_surface_dataset_path = (
        '../../mnt/local/data/kalexu97/topOpt/ellipse_sdf_surface_dataset_smf22'
    )  # without .csv!!
    df, points_df = generate_ellipse_sdf_surface_dataset(
        num_ellipse=args.num_surface_shapes,
        points_per_ellipse=args.points_per_surface_shape,
        smooth_factor=int(args.smooth_factor*1.1),
        filename=ellipse_surface_dataset_path
    )

    quadrangle_surface_dataset_path = (
        '../../mnt/local/data/kalexu97/topOpt/rounded_quadrangle_sdf_surface_dataset_smf20'
    )  # without .csv!!
    df, points_df = generate_rounded_quadrangle_sdf_surface_dataset(
        num_quadrangle=args.num_surface_shapes,
        points_per_quadrangle=args.points_per_surface_shape,
        smooth_factor=args.smooth_factor,
        filename=quadrangle_surface_dataset_path
    )

    triangle_surface_dataset_path = (
        '../../mnt/local/data/kalexu97/topOpt/rounded_triangle_sdf_surface_dataset_smf20'
    )  # without .csv!!
    df, points_df = generate_rounded_triangle_sdf_surface_dataset(
        num_triangle=args.num_surface_shapes,
        points_per_triangle=args.points_per_surface_shape,
        smooth_factor=args.smooth_factor,
        filename=triangle_surface_dataset_path
    )
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('--num_shapes', type=int, default=5000, help='Number of shapes to generate')
    parser.add_argument('--points_per_shape', type=int, default=1000, help='Number of points per shape')
    parser.add_argument('--num_surface_shapes', type=int, default=300, help='Number of surface shapes to generate')
    parser.add_argument('--points_per_surface_shape', type=int, default=1225, help='Number of points per surface shape')
    parser.add_argument('--smooth_factor', type=int, default=20, help='Smooth factor')
    parser.add_argument('--arc_ratio', type=float, default=0.5, help='Arc ratio')
    args = parser.parse_args()
    main(args)