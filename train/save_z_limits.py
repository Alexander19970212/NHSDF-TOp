import sys
import os
import torch
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.sdf_models import (
    LitSdfAE_MINE,
    AE,
    VAE,
    MMD_VAE,
    AE_DeepSDF,
    VAE_DeepSDF,
    MMD_VAE_DeepSDF
)
from datasets.SDF_dataset import SdfDataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="Save Z Limits Script")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--configs_dir",
        type=str,
        required=True,
        help="Path to the configurations directory."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Path to the models directory."
    )
    return parser.parse_args()

def load_model(configs_dir, models_dir, config_name):
    models = {
        'AE_DeepSDF': AE_DeepSDF,
        'AE': AE, 
        'VAE': VAE,
        'VAE_DeepSDF': VAE_DeepSDF,
        'MMD_VAE': MMD_VAE,
        'MMD_VAE_DeepSDF': MMD_VAE_DeepSDF
    }

    # Load configuration from YAML file
    config_path = os.path.join(configs_dir, f"{config_name}.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize model
    model_type = config['model']['type']
    model_params = config['model']['params']
    model_params['input_dim'] = 17  # train_dataset.feature_dim
    model = models[model_type](**model_params)

    # Load pre-trained weights
    saved_model_path = os.path.join(models_dir, f"uba_frst_{config_name}_full.pt")
    state_dict = torch.load(saved_model_path)
    new_state_dict = model.state_dict()

    # Update the new_state_dict with the loaded state_dict, ignoring size mismatches
    for key in state_dict:
        if key in new_state_dict and state_dict[key].size() == new_state_dict[key].size():
            new_state_dict[key] = state_dict[key]

    model.load_state_dict(state_dict)
    model.eval()
    return model, config_name

def investigate_latent_space(model, dataloader, stats_dir, config_name):
    """Visualize the latent space"""

    # Create stats directory if it doesn't exist
    if stats_dir is not None:
        os.makedirs(stats_dir, exist_ok=True)

    latent_vectors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            output = model(batch[0])
            latent_vectors.append(output['z'])
                
    latent_vectors = torch.cat(latent_vectors, dim=0)
    latent_vectors = latent_vectors.cpu().numpy()

    latent_mins = np.min(latent_vectors, axis=0)
    latent_maxs = np.max(latent_vectors, axis=0)

    if stats_dir is not None and config_name is not None:
        np.savez(
            os.path.join(stats_dir, f"{config_name}_full_stats.npz"),
            latent_mins=latent_mins,
            latent_maxs=latent_maxs
        )

def main():
    args = parse_arguments()

    # Add necessary paths
    sys.path.append(os.path.abspath('..'))
    sys.path.append(os.path.abspath('NN_TopOpt'))

    # Initialize dataset
    dataset_test_files = [
        os.path.join(args.dataset_path, 'ellipse_sdf_dataset_smf22_arc_ratio_500.csv'),
        os.path.join(args.dataset_path, 'triangle_sdf_dataset_smf20_arc_ratio_500.csv'),
        os.path.join(args.dataset_path, 'quadrangle_sdf_dataset_smf20_arc_ratio_500.csv')
    ]
    
    test_dataset = SdfDataset(dataset_test_files, exclude_ellipse=False)
    
    # Create DataLoader with shuffling
    batch_size = 64
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=15
    )

    # Load model
    config_name = 'MMD_VAE_DeepSDF'
    model, config_name = load_model(args.configs_dir, args.models_dir, config_name)

    # Investigate latent space
    stats_dir = '../z_limits'
    investigate_latent_space(model, test_loader, stats_dir=stats_dir, config_name=config_name)

if __name__ == "__main__":
    main()