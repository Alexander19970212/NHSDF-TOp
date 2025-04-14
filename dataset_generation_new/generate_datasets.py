import os

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
from triangle_sdf import generate_traingle_reconstruction_dataset
from quadrangle_sdf import generate_quadrangle_reconstruction_dataset
from ellipse_sdf import generate_ellipse_reconstruction_dataset


from ellipse_sdf import generate_ellipse_vae_dataset
from quadrangle_sdf import generate_quadrangle_vae_dataset
from triangle_sdf import generate_triangle_vae_dataset
from utils_generation import plot_vae_dataset

import argparse

def main(args):
    root_path = args.root_path
    num_golden_quadrangle = 3
    num_golden_ellipse = 3
    os.makedirs(root_path, exist_ok=True)
    smooth_factor = 20
    suffix = 'Bprec'


    #################### train dataset ####################
    num_samples = 15000

    # dataset_path = f'{root_path}/ellipse_sdf_dataset_smf{smooth_factor}_arc_ratio_{num_samples}_{suffix}.csv'
    # df = generate_ellipse_sdf_dataset(num_ellipse=num_samples,
    #                                   points_per_ellipse=1000,
    #                                   smooth_factor=smooth_factor,
    #                                   filename=dataset_path,
    #                                   num_golden_ellipse=num_golden_ellipse)

    # dataset_path = f'{root_path}/triangle_sdf_dataset_smf{smooth_factor}_arc_ratio_{num_samples}_{suffix}.csv'
    # df = generate_rounded_triangle_sdf_dataset(num_triangle=num_samples, points_per_triangle=1000, smooth_factor=smooth_factor, filename=dataset_path)

    dataset_path = f'{root_path}/quadrangle_sdf_dataset_smf{smooth_factor}_arc_ratio_{num_samples}_{suffix}.csv'
    df = generate_rounded_quadrangle_sdf_dataset(num_quadrangle=num_samples,
                                                 points_per_quadrangle=1000,
                                                 smooth_factor=smooth_factor,
                                                 filename=dataset_path,
                                                 num_golden_quadrangle=num_golden_quadrangle)

    # # #################### surface dataset ####################

    # dataset_path = f'{root_path}/ellipse_sdf_surface_dataset_smf{smooth_factor}_150_{suffix}.csv'
    # df, points_df = generate_ellipse_sdf_surface_dataset(num_ellipse=150, points_per_ellipse=1225, smooth_factor=smooth_factor, filename=dataset_path)

    dataset_path = f'{root_path}/quadrangle_sdf_surface_dataset_smf{smooth_factor}_150_{suffix}.csv'
    df, points_df = generate_rounded_quadrangle_sdf_surface_dataset(num_quadrangle=150, points_per_quadrangle=1225, smooth_factor=smooth_factor, filename=dataset_path)

    # dataset_path = f'{root_path}/triangle_sdf_surface_dataset_smf{smooth_factor}_150_{suffix}.csv'
    # df, points_df = generate_rounded_triangle_sdf_surface_dataset(num_triangle=150, points_per_triangle=1225, smooth_factor=smooth_factor, filename=dataset_path)

    # #################### test dataset ####################

    # dataset_path = f'{root_path}/ellipse_sdf_dataset_smf{smooth_factor}_arc_ratio_500_test_{suffix}.csv'
    # df = generate_ellipse_sdf_dataset(num_ellipse=500,
    #                                   points_per_ellipse=1000,
    #                                   smooth_factor=smooth_factor,
    #                                   filename=dataset_path,
    #                                   num_golden_ellipse=num_golden_ellipse)

    # dataset_path = f'{root_path}/triangle_sdf_dataset_smf{smooth_factor}_arc_ratio_500_test_{suffix}.csv'
    # df = generate_rounded_triangle_sdf_dataset(num_triangle=500, points_per_triangle=1000, smooth_factor=smooth_factor, filename=dataset_path)

    dataset_path = f'{root_path}/quadrangle_sdf_dataset_smf{smooth_factor}_arc_ratio_500_test_{suffix}.csv'
    df = generate_rounded_quadrangle_sdf_dataset(num_quadrangle=500,
                                                 points_per_quadrangle=1000,
                                                 smooth_factor=smooth_factor,
                                                 filename=dataset_path,
                                                 num_golden_quadrangle=num_golden_quadrangle)

    # #################### radius dataset ####################

    # dataset_path = f'{root_path}/triangle_sdf_dataset_smf40_radius_sample_100'
    # df = generate_traingle_random_radius_dataset(num_triangle=100, sample_per_triangle=100, smooth_factor=40, filename=dataset_path)

    # dataset_path = f'{root_path}/quadrangle_sdf_dataset_smf40_radius_sample_100'
    # df = generate_quadrangle_random_radius_dataset(num_quadrangle=100, sample_per_quadrangle=100, smooth_factor=40, filename=dataset_path)

    #################### reconstruction dataset ####################

    n_features_per_shape = 5000000

    # training dataset
    # dataset_path = f'{root_path}/triangle_reconstruction_dataset_train_{suffix}'
    # df = generate_traingle_reconstruction_dataset(num_triangle=n_features_per_shape, smooth_factor=smooth_factor, filename=dataset_path)

    dataset_path = f'{root_path}/quadrangle_reconstruction_dataset_train_{suffix}'
    df = generate_quadrangle_reconstruction_dataset(num_quadrangle=n_features_per_shape,
                                                    smooth_factor=smooth_factor,
                                                    filename=dataset_path,
                                                    num_golden_quadrangle=num_golden_quadrangle)

    # dataset_path = f'{root_path}/ellipse_reconstruction_dataset_train_{suffix}'
    # df = generate_ellipse_reconstruction_dataset(num_ellipse=n_features_per_shape,
    #                                              smooth_factor=smooth_factor,
    #                                              filename=dataset_path,
    #                                              num_golden_ellipse=num_golden_ellipse)

    n_features_per_shape = 10000

    # test dataset
    # dataset_path = f'{root_path}/triangle_reconstruction_dataset_test_{suffix}'
    # df = generate_traingle_reconstruction_dataset(num_triangle=n_features_per_shape, smooth_factor=smooth_factor, filename=dataset_path)

    dataset_path = f'{root_path}/quadrangle_reconstruction_dataset_test_{suffix}'
    df = generate_quadrangle_reconstruction_dataset(num_quadrangle=n_features_per_shape,
                                                    smooth_factor=smooth_factor,
                                                    filename=dataset_path,
                                                    num_golden_quadrangle=num_golden_quadrangle)

    # dataset_path = f'{root_path}/ellipse_reconstruction_dataset_test_{suffix}'
    # df = generate_ellipse_reconstruction_dataset(num_ellipse=n_features_per_shape,
    #                                              smooth_factor=smooth_factor,
    #                                              filename=dataset_path,
    #                                              num_golden_ellipse=num_golden_ellipse)

def generate_datasets_for_conv_vae(args):
    root_path = args.root_path
    dataset_name = args.dataset_name

    image_size = 64
    n_golden_gf = 2
    n_sample_per_gf_train = 100000
    n_sample_per_gf_test = 10000

    smooth_factor = 15

    dataset_dir = f'{root_path}/{dataset_name}'
    os.makedirs(dataset_dir, exist_ok=True)

    train_dataset_dir = f'{dataset_dir}/train'
    os.makedirs(train_dataset_dir, exist_ok=True)

    test_dataset_dir = f'{dataset_dir}/test'
    os.makedirs(test_dataset_dir, exist_ok=True)

    generate_ellipse_vae_dataset(num_ellipse=n_sample_per_gf_train,
                                smooth_factor=smooth_factor,
                                dataset_dir=train_dataset_dir,
                                num_golden_ellipse=n_golden_gf,
                                image_size=image_size)

    generate_quadrangle_vae_dataset(num_quadrangle=n_sample_per_gf_train,
                                smooth_factor=smooth_factor,
                                dataset_dir=train_dataset_dir,
                                num_golden_quadrangle=n_golden_gf,
                                image_size=image_size)
    
    generate_triangle_vae_dataset(num_triangle=n_sample_per_gf_train,
                                smooth_factor=smooth_factor,
                                dataset_dir=train_dataset_dir,
                                num_golden_triangle=n_golden_gf,
                                image_size=image_size)
    
    generate_ellipse_vae_dataset(num_ellipse=n_sample_per_gf_test,
                                smooth_factor=smooth_factor,
                                dataset_dir=test_dataset_dir,
                                num_golden_ellipse=n_golden_gf,
                                image_size=image_size)

    generate_quadrangle_vae_dataset(num_quadrangle=n_sample_per_gf_test,
                                smooth_factor=smooth_factor,
                                dataset_dir=test_dataset_dir,
                                num_golden_quadrangle=n_golden_gf,
                                image_size=image_size)
    
    generate_triangle_vae_dataset(num_triangle=n_sample_per_gf_test,
                                smooth_factor=smooth_factor,
                                dataset_dir=test_dataset_dir,
                                num_golden_triangle=n_golden_gf,
                                image_size=image_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate datasets for VAE training')
    parser.add_argument('--root_path', type=str, default='../shape_datasets', help='Root path for the datasets')
    parser.add_argument('--dataset_name', type=str, default='conv_vae', help='Dataset name')
    parser.add_argument('--vae_type', type=str, default='default', help='VAE type')

    args = parser.parse_args()
    if args.vae_type == 'conv':
        generate_datasets_for_conv_vae(args)
    else:
        main(args)
