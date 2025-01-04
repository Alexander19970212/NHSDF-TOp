from ellipse_sdf import generate_ellipse_sdf_dataset, plot_ellipse_sdf_dataset
from quadrangle_sdf import generate_quadrangle_sdf_dataset, plot_quadrangle_sdf_dataset
from triangle_sdf import generate_triangle_sdf_dataset, plot_triangle_sdf_dataset
from triangle_sdf import generate_rounded_triangle_sdf_dataset, plot_triangle_sdf_dataset
from quadrangle_sdf import generate_rounded_quadrangle_sdf_dataset, plot_quadrangle_sdf_dataset
from ellipse_sdf import generate_ellipse_sdf_surface_dataset, plot_ellipse_sdf_dataset
from utils_generation import plot_sample_from_df
from quadrangle_sdf import generate_rounded_quadrangle_sdf_surface_dataset, plot_quadrangle_sdf_dataset
from triangle_sdf import generate_rounded_triangle_sdf_surface_dataset, plot_triangle_sdf_dataset
from triangle_sdf import generate_traingle_random_radius_dataset
from quadrangle_sdf import generate_quadrangle_random_radius_dataset

import argparse

def main(args):
    root_path = args.root_path

    #################### train dataset ####################

    # dataset_path = f'{root_path}/ellipse_sdf_dataset_smf22_arc_ratio_5000.csv'
    # # dataset_path = '../../mnt/local/data/kalexu97/topOpt/ellipse_sdf_dataset_smf22_arc_ratio.csv'
    # df = generate_ellipse_sdf_dataset(num_ellipse=5000, points_per_ellipse=1000, smooth_factor=22, filename=dataset_path)

    # dataset_path = f'{root_path}/triangle_sdf_dataset_smf20_arc_ratio_5000.csv'
    # # dataset_path = '../../mnt/local/data/kalexu97/topOpt/rounded_triangle_sdf_dataset_smf20_arc_ratio.csv'
    # df = generate_rounded_triangle_sdf_dataset(num_triangle=5000, points_per_triangle=1000, smooth_factor=20, filename=dataset_path)

    # dataset_path = f'{root_path}/quadrangle_sdf_dataset_smf20_arc_ratio_5000.csv'
    # # dataset_path = '../../mnt/local/data/kalexu97/topOpt/rounded_quadrangle_sdf_dataset_smf20_arc_ratio.csv'
    # df = generate_rounded_quadrangle_sdf_dataset(num_quadrangle=5000, points_per_quadrangle=1000, smooth_factor=20, filename=dataset_path)

    # #################### surface dataset ####################

    # dataset_path = f'{root_path}/ellipse_sdf_surface_dataset_smf22_150.csv'
    # # dataset_path = '../../mnt/local/data/kalexu97/topOpt/ellipse_sdf_surface_dataset_smf22' # without .csv!!
    # df, points_df = generate_ellipse_sdf_surface_dataset(num_ellipse=150, points_per_ellipse=1225, smooth_factor=22, filename=dataset_path)

    # dataset_path = f'{root_path}/quadrangle_sdf_surface_dataset_smf20_150.csv'
    # # dataset_path = '../../mnt/local/data/kalexu97/topOpt/rounded_quadrangle_sdf_surface_dataset_smf20' # without .csv!!
    # df, points_df = generate_rounded_quadrangle_sdf_surface_dataset(num_quadrangle=150, points_per_quadrangle=1225, smooth_factor=20, filename=dataset_path)

    # dataset_path = f'{root_path}/triangle_sdf_surface_dataset_smf20_150.csv'
    # # dataset_path = '../../mnt/local/data/kalexu97/topOpt/rounded_triangle_sdf_surface_dataset_smf20' # without .csv !!
    # df, points_df = generate_rounded_triangle_sdf_surface_dataset(num_triangle=150, points_per_triangle=1225, smooth_factor=20, filename=dataset_path)

    #################### test dataset ####################

    dataset_path = f'{root_path}/ellipse_sdf_dataset_smf22_arc_ratio_500_test.csv'
    # dataset_path = '../../mnt/local/data/kalexu97/topOpt/ellipse_sdf_dataset_smf22_arc_ratio.csv'
    df = generate_ellipse_sdf_dataset(num_ellipse=500, points_per_ellipse=1000, smooth_factor=22, filename=dataset_path)

    dataset_path = f'{root_path}/triangle_sdf_dataset_smf20_arc_ratio_500_test.csv'
    # dataset_path = '../../mnt/local/data/kalexu97/topOpt/rounded_triangle_sdf_dataset_smf20_arc_ratio.csv'
    df = generate_rounded_triangle_sdf_dataset(num_triangle=500, points_per_triangle=1000, smooth_factor=20, filename=dataset_path)

    dataset_path = f'{root_path}/quadrangle_sdf_dataset_smf20_arc_ratio_500_test.csv'
    # dataset_path = '../../mnt/local/data/kalexu97/topOpt/rounded_quadrangle_sdf_dataset_smf20_arc_ratio.csv'
    df = generate_rounded_quadrangle_sdf_dataset(num_quadrangle=500, points_per_quadrangle=1000, smooth_factor=20, filename=dataset_path)

    #################### radius dataset ####################

    dataset_path = f'{root_path}/triangle_sdf_dataset_smf40_radius_sample_100'
    df = generate_traingle_random_radius_dataset(num_triangle=100, sample_per_triangle=100, smooth_factor=40, filename=dataset_path)

    dataset_path = f'{root_path}/quadrangle_sdf_dataset_smf40_radius_sample_100'
    df = generate_quadrangle_random_radius_dataset(num_quadrangle=100, sample_per_quadrangle=100, smooth_factor=40, filename=dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate datasets for VAE training')
    parser.add_argument('--root_path', type=str, default='../shape_datasets', help='Root path for the datasets')
    args = parser.parse_args()
    main(args)
