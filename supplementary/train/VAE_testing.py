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

from models.sdf_models import LitSdfAE
from models.sdf_models import AE
from models.sdf_models import AE_DeepSDF, AE_DeepSDF_explicit_radius
from datasets.SDF_dataset import SdfDataset, SdfDatasetSurface, collate_fn_surface
from datasets.SDF_dataset import RadiusDataset
import argparse
import json

# Add the parent directory of NN_TopOpt to the system path
sys.path.append(os.path.abspath('NN_TopOpt'))


def main(args):
    root_path = 'shape_datasets'

    dataset_train_files = [f'{root_path}/ellipse_sdf_dataset_smf22_arc_ratio_5000.csv',
                    f'{root_path}/triangle_sdf_dataset_smf20_arc_ratio_5000.csv', 
                    f'{root_path}/quadrangle_sdf_dataset_smf20_arc_ratio_5000.csv']
    
    dataset_test_files = [f'{root_path}/ellipse_sdf_dataset_smf22_arc_ratio_500_test.csv',
                 f'{root_path}/triangle_sdf_dataset_smf20_arc_ratio_500_test.csv', 
                 f'{root_path}/quadrangle_sdf_dataset_smf20_arc_ratio_500_test.csv']
    
    surface_files = [f'{root_path}/ellipse_sdf_surface_dataset_smf22_150.csv',
                 f'{root_path}/triangle_sdf_surface_dataset_smf20_150.csv',
                 f'{root_path}/quadrangle_sdf_surface_dataset_smf20_150.csv']
    
    radius_samples_files = [f'{root_path}/triangle_sdf_dataset_smf40_radius_sample_100.csv',
                            f'{root_path}/quadrangle_sdf_dataset_smf40_radius_sample_100.csv']

    # dataset_files = ['shape_datasets/ellipse_sdf_dataset_onlMove.csv',
    #                  'shape_datasets/triangle_sdf_dataset_test.csv', 
    #                  'shape_datasets/quadrangle_sdf_dataset_test.csv']
    
    # surface_files = ['shape_datasets/ellipse_sdf_surface_dataset_test',
    #                  'shape_datasets/triangle_sdf_surface_dataset_test',
    #                  'shape_datasets/quadrangle_sdf_surface_dataset_test']

    train_dataset = SdfDataset(dataset_train_files, exclude_ellipse=False)
    test_dataset = SdfDataset(dataset_test_files, exclude_ellipse=False)
    surface_dataset = SdfDatasetSurface(surface_files, cut_value=False)
    radius_samples_dataset = RadiusDataset(radius_samples_files)

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
        batch_size=batch_size,
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
   
    radius_samples_loader = torch.utils.data.DataLoader(
        radius_samples_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=15
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Surface test set size: {len(surface_test_loader)}")
    print(f"Radius samples set size: {len(radius_samples_loader)}")

    #################################################

    # load model
    checkpoint_path = f'model_weights/{args.run_name}.ckpt'

    vae_model = AE_DeepSDF_explicit_radius(
        input_dim=test_dataset.feature_dim, 
        latent_dim=9, 
        hidden_dim=128, 
        rad_latent_dim=2,
        rad_loss_weight=0.1,
        orthogonality_loss_weight=0.1,
        regularization='l2',   # Use 'l1', 'l2', or None
        reg_weight=1e-3        # Adjust the weight as needed
    )

    vae_trainer = LitSdfAE.load_from_checkpoint(checkpoint_path, vae_model=vae_model, strict=True)

    # Training setup
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=1,
        logger=TensorBoardLogger(name='VAEi', save_dir='./logs', default_hp_metric=False, version=args.run_name),
        callbacks=[
            callbacks.ModelCheckpoint(
                monitor='val_radius_sum_loss',
                mode='min',
                save_top_k=1,
                filename='best-model-{epoch:02d}-{val_radius_sum_loss:.2f}'
            ),
            callbacks.EarlyStopping(
                monitor='val_radius_sum_loss',   
                patience=10,
                mode='min'
            )
        ]
    )

    metrics = trainer.validate(vae_trainer, dataloaders=[test_loader, surface_test_loader, radius_samples_loader])

    combined_metrics = {k: v for d in metrics for k, v in d.items()}

    # Save metrics into a JSON file with run_name
    metrics_filename = 'src/metrics.json'
    try:
        with open(metrics_filename, 'r') as f:
            all_metrics = json.load(f)
    except FileNotFoundError:
        all_metrics = {}

    all_metrics[args.run_name] = combined_metrics

    with open(metrics_filename, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('--max_epochs', type=int, default=1, help='Maximum number of epochs for training')
    parser.add_argument('--run_name', type=str, default='uba_NTM_F_reg5em3', help='Name of the run')
    args = parser.parse_args()
    main(args)
