import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from lightning.pytorch import Trainer, seed_everything, callbacks
from lightning.pytorch.loggers import TensorBoardLogger

# Add the parent directory of NN_TopOpt to the system path
sys.path.append(os.path.abspath('..'))

from models.sdf_models import LitSdfAE_MINE
from models.sdf_models import AE, VAE, MMD_VAE
from models.sdf_models import AE_DeepSDF, VAE_DeepSDF, MMD_VAE_DeepSDF
from datasets.SDF_dataset import SdfDataset, SdfDatasetSurface, collate_fn_surface
from datasets.SDF_dataset import RadiusDataset
from datasets.SDF_dataset import ReconstructionDataset
from models.sdf_models import LitRecon_MINE

from models.sdf_models import LitSdfAE_Reconstruction

from vizualization_utils import plot_latent_space, plot_latent_space_radius_sum, plot_sdf_surface, plot_sdf_transition_triangle
from vizualization_utils import get_latent_subspaces

import json

import argparse
# Enable anomaly detection to help find where NaN/Inf values originate
torch.autograd.set_detect_anomaly(True)

models = {'AE_DeepSDF': AE_DeepSDF,
            'AE': AE, 
            'VAE': VAE,
            'VAE_DeepSDF': VAE_DeepSDF,
            'MMD_VAE': MMD_VAE,
            'MMD_VAE_DeepSDF': MMD_VAE_DeepSDF}

import yaml


# Add the parent directory of NN_TopOpt to the system path
sys.path.append(os.path.abspath('..'))

arch_dirs = {'VAE_DeepSDF': 'VAEs',
            'AE_DeepSDF': 'AEs',
            'VAE': 'VAEs',
            'AE': 'AEs',
            'MMD_VAE': 'MMD_VAEs',
            'MMD_VAE_DeepSDF': 'MMD_VAEs'}

def main(args):
    root_path = '../shape_datasets'
    models_dir = '../model_weights'

    config_name = args.config_name
    strategy = args.strategy
    quadrangle_index = args.quadrangle_index
    triangle_index = args.triangle_index
    ellipse_index = args.ellipse_index

    configs_dir = f'../configs/NN_sdf_experiments/final_experiments/{arch_dirs[config_name]}'
    saved_model_path = f'{models_dir}/{strategy}_{config_name}_full.pt'

    save_plot_dir = f'../src/{strategy}_{config_name}'
    os.makedirs(save_plot_dir, exist_ok=True)

    filename_latents = f'{save_plot_dir}/latents.png'
    filename_rec_geometry = f'{save_plot_dir}/reconstruction_geometry.png'
    filename_hv_border = f'{save_plot_dir}/reconstruction_Hv_border.png'

    filename_sdf_quadrangle = f'{save_plot_dir}/sdf_quadrangle.png'
    filename_sdf_triangle = f'{save_plot_dir}/sdf_triangle.png'
    filename_sdf_ellipse = f'{save_plot_dir}/sdf_ellipse.png'

    dataset_test_files = [f'{root_path}/ellipse_reconstruction_dataset_test.csv',
                        f'{root_path}/triangle_reconstruction_dataset_test.csv',
                        f'{root_path}/quadrangle_reconstruction_dataset_test.csv',
                    ]

    test_dataset = ReconstructionDataset(dataset_test_files)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1024,
        shuffle=False,  # No need to shuffle test data
        num_workers=15
    )

    print(f"Test set size: {len(test_dataset)}")

    # Load configuration from YAML file
    with open(f'{configs_dir}/{config_name}.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize VAE model
    model_params = config['model']['params']
    model_params['input_dim'] = 17 # train_dataset.feature_dim

    print(f"Loading model {strategy}_{config_name}")
    vae_model = models[config['model']['type']](**model_params)

    # Load pre-trained weights for the model
    print(f"Loading model weights from {saved_model_path}")
    state_dict = torch.load(saved_model_path)
    vae_model.load_state_dict(state_dict)

    ########## temporary ##########
    searching_points_path = f'{save_plot_dir}/searching_points.json'
    searching_points = {
        "tsne_coords": [
            [-52, 45],
            [5, -120],
            [30, -70],
            [30, -70],
            [70, 60]
        ],
        "axes_positions": [
            [-140, 50],
            [-120, -120],
            [80, -100],
            [40, -120],
            [80, 100]
        ],
        "gf_types":[ # n: doesn't matter, c: circle, t: triangle, q: quadrangle
            "n",
            "n",
            "t",
            "q",
            "n"
        ]
    }
    json.dump(searching_points, open(searching_points_path, 'w'))
    ########## temporary ##########
    print(f"Plotting latent space ...")
    plot_latent_space(vae_model, test_loader, filename=filename_latents)

    print(f"Getting latent subspaces ...")
    triangle_latent_vectors, quadrangle_latent_vectors, ellipse_latent_vectors = get_latent_subspaces(
        vae_model, test_loader
    )

    z_triangle = triangle_latent_vectors[triangle_index]
    z_quadrangle = quadrangle_latent_vectors[quadrangle_index]
    z_ellipse = ellipse_latent_vectors[ellipse_index]

    print(f"Plotting SDF surfaces ...")
    plot_sdf_surface(vae_model, z_quadrangle,
                     countur=False, filename=filename_sdf_quadrangle)
    plot_sdf_surface(vae_model, z_triangle,
                     countur=False, filename=filename_sdf_triangle)
    plot_sdf_surface(vae_model, z_ellipse,
                     countur=False, filename=filename_sdf_ellipse)

    print(f"Plotting SDF transition ...")
    plot_sdf_transition_triangle(vae_model,
                                z_triangle,
                                z_quadrangle,
                                z_ellipse,
                                num_steps=6,
                                filename=filename_rec_geometry,
                                plot_geometry=True)

    plot_sdf_transition_triangle(vae_model,
                                z_triangle,
                                z_quadrangle,
                                z_ellipse,
                                num_steps=6,
                                filename=filename_hv_border,
                                plot_geometry=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='VAE_DeepSDF')
    parser.add_argument('--strategy', type=str, default='frst')
    parser.add_argument('--quadrangle_index', type=int, default=1)
    parser.add_argument('--triangle_index', type=int, default=0)
    parser.add_argument('--ellipse_index', type=int, default=4)

    args = parser.parse_args()
    main(args)