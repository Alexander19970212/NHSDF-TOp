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

# Add parent directory to path since script is run from parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets'))

from models.sdf_models import LitHvDecoderGlobal, LitSdfAE_MINE, LitSdfAE_HvDecoder
from models.sdf_models import AE
from models.sdf_models import AE_DeepSDF
from models.sdf_models import VAE
from models.sdf_models import VAE_DeepSDF
from models.sdf_models import MMD_VAE
from models.sdf_models import MMD_VAE_DeepSDF
try:
    from SDF_dataset import SdfDataset, SdfDatasetSurface, collate_fn_surface, RadiusDataset
except ImportError:
    from datasets.SDF_dataset import SdfDataset, SdfDatasetSurface, collate_fn_surface, RadiusDataset
import argparse
import yaml
import json

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # safer to use before loading lightning.gpu

# Clean CUDA cache
torch.cuda.empty_cache()


# Set all random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed_everything(seed)  # Lightning seed


# Add the parent directory of NN_TopOpt to the system path
sys.path.append(os.path.abspath('NN_TopOpt'))

models = {'AE_DeepSDF': AE_DeepSDF,
          'AE': AE, 
          'VAE': VAE,
          'VAE_DeepSDF': VAE_DeepSDF,
          'MMD_VAE': MMD_VAE,
          'MMD_VAE_DeepSDF': MMD_VAE_DeepSDF}

trainers = {'LitSdfAE': LitHvDecoderGlobal,
            'LitSdfAE_MINE': LitSdfAE_MINE}


def main(args):
    dataset_path = args.dataset_path
    configs_dir = args.config_dir
    config_name = args.config_name
    models_dir = args.model_dir

    run_name = f'uba_{config_name}'

    saved_model_path = f'{models_dir}/{config_name}_ReconDecGlobal.pt'

    dataset_train_files = [f'{dataset_path}/ellipse_sdf_dataset_smf22_arc_ratio_5000.csv',
                    f'{dataset_path}/triangle_sdf_dataset_smf20_arc_ratio_5000.csv', 
                    f'{dataset_path}/quadrangle_sdf_dataset_smf20_arc_ratio_5000.csv']
    
    dataset_test_files = [f'{dataset_path}/ellipse_sdf_dataset_smf22_arc_ratio_500_test.csv',
                 f'{dataset_path}/triangle_sdf_dataset_smf20_arc_ratio_500_test.csv', 
                 f'{dataset_path}/quadrangle_sdf_dataset_smf20_arc_ratio_500_test.csv']
    
    surface_files = [f'{dataset_path}/ellipse_sdf_surface_dataset_smf22_150.csv',
                 f'{dataset_path}/triangle_sdf_surface_dataset_smf20_150.csv',
                 f'{dataset_path}/quadrangle_sdf_surface_dataset_smf20_150.csv']
    
    radius_samples_files = [f'{dataset_path}/triangle_sdf_dataset_smf40_radius_sample_100.csv',
                            f'{dataset_path}/quadrangle_sdf_dataset_smf40_radius_sample_100.csv']

    # dataset_files = ['shape_datasets/ellipse_sdf_dataset_onlMove.csv',
    #                  'shape_datasets/triangle_sdf_dataset_test.csv', 
    #                  'shape_datasets/quadrangle_sdf_dataset_test.csv']
    
    # surface_files = ['shape_datasets/ellipse_sdf_surface_dataset_test',
    #                  'shape_datasets/triangle_sdf_surface_dataset_test',
    #                  'shape_datasets/quadrangle_sdf_surface_dataset_test']

    train_dataset = SdfDataset(dataset_train_files, exclude_ellipse=False)
    test_dataset = SdfDataset(dataset_test_files, exclude_ellipse=False)
    surface_dataset = SdfDatasetSurface(surface_files, cut_value=False)
    # radius_samples_dataset = RadiusDataset(radius_samples_files)

    # Create DataLoaders with shuffling
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Enable shuffling for training data
        num_workers=15
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1024,
        shuffle=False,  # No need to shuffle test data
        num_workers=15
    )

    surface_test_loader = torch.utils.data.DataLoader(
        surface_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=15,
        collate_fn=collate_fn_surface
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Surface test set size: {len(surface_test_loader)}")

    #################################################

    MAX_EPOCHS = args.max_epochs
    MAX_STEPS = MAX_EPOCHS * len(train_loader)

    # Training setup
    trainer = Trainer(
        max_epochs=MAX_EPOCHS, # the first epoch for training all model, the second one for training rec decoder
        accelerator='auto',
        devices=1,
        logger=TensorBoardLogger(
            name='HvDecSeparate', 
            save_dir='./logs', 
            default_hp_metric=False, 
            version=run_name
        ),
        callbacks=[
            callbacks.ModelCheckpoint(
                monitor='val_total_loss/dataloader_idx_0',
                mode='min',
                save_top_k=1,
                filename='best-model-{epoch:02d}-{val_total_loss:.2f}'
            ),
            callbacks.EarlyStopping(
                monitor='val_total_loss/dataloader_idx_0',
                patience=10,
                mode='min'
            ) #,
            # FirstEvalCallback()
        ],
        check_val_every_n_epoch=None,  # Disable validation every epoch
        val_check_interval=50000  # Perform validation every 2000 training steps
    )

    
    # Load configuration from YAML file
    with open(f'{configs_dir}/{config_name}.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize VAE model
    model_params = config['model']['params']
    model_params['input_dim'] = test_dataset.feature_dim
    vae_model = models[config['model']['type']](**model_params)

    # Load pre-trained weights for the model
    state_dict = torch.load(saved_model_path)
    vae_model.load_state_dict(state_dict)

    # Initialize VAE trainer
    trainer_params = config['trainer']['params']
    trainer_params['vae_model'] = vae_model
    trainer_params['max_steps'] = MAX_STEPS
    # vae_trainer = trainers[config['trainer']['type']](**trainer_params)
    vae_trainer = LitSdfAE_HvDecoder(**trainer_params)

    # Train the model
    trainer.validate(vae_trainer, dataloaders=[test_loader, surface_test_loader])
    trainer.fit(vae_trainer, train_loader, val_dataloaders=[test_loader, surface_test_loader])
    final_metrics = trainer.validate(vae_trainer, dataloaders=[test_loader, surface_test_loader])

    # Save model weights
    checkpoint_path = f'{models_dir}/{run_name}_full.ckpt'
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model weights saved to {checkpoint_path}")

    # Save just the model weights
    model_weights_path = f'{models_dir}/{run_name}_full.pt'
    torch.save(vae_model.state_dict(), model_weights_path)
    print(f"Model weights saved to {model_weights_path}")

    # Save metrics into a JSON file with run_name
    combined_metrics = {k: v for d in final_metrics for k, v in d.items()}

    # Save metrics into a JSON file with run_name
    metrics_filename = args.metrics_file
    try:
        with open(metrics_filename, 'r') as f:
            all_metrics = json.load(f)
    except FileNotFoundError:
        all_metrics = {}

    all_metrics[run_name] = combined_metrics

    with open(metrics_filename, 'w') as f:
        json.dump(all_metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('--max_epochs', type=int, default=1, help='Maximum number of epochs for training')
    # parser.add_argument('--run_name', type=str, default='uba_NTM_F_reg5em3', help='Name of the run')
    parser.add_argument('--model_dir', type=str, default='model_weights', help='Path to the model directory')
    parser.add_argument('--dataset_path', type=str, default='shape_datasets', help='Path to the dataset')
    parser.add_argument('--config_dir', type=str, default='configs/NN_sdf_experiments/architectures', help='Path to the config directory')
    parser.add_argument('--config_name', type=str, default='AE_DeepSDF', help='Name of the config')
    parser.add_argument('--metrics_file', type=str, default='src/metrics.json', help='Path to the metrics file')
    args = parser.parse_args()
    main(args)
