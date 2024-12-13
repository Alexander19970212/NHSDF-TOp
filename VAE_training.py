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

from models.sdf_models import LitSdfVAE, VAE
from datasets.SDF_dataset import SdfDataset, SdfDatasetSurface, collate_fn_surface
import argparse

def main(args):
    dataset_files = ['../mnt/local/data/kalexu97/topOpt/ellipse_sdf_dataset.csv',
                     '../mnt/local/data/kalexu97/topOpt/rounded_triangle_sdf_dataset.csv', 
                     '../mnt/local/data/kalexu97/topOpt/rounded_quadrangle_sdf_dataset.csv']

    surface_files = ['../mnt/local/data/kalexu97/topOpt/ellipse_sdf_surface_dataset',
                     '../mnt/local/data/kalexu97/topOpt/rounded_triangle_sdf_surface_dataset', 
                     '../mnt/local/data/kalexu97/topOpt/rounded_quadrangle_sdf_surface_dataset']

    # dataset_files = ['shape_datasets/ellipse_sdf_dataset_onlMove.csv',
    #                  'shape_datasets/triangle_sdf_dataset_test.csv', 
    #                  'shape_datasets/quadrangle_sdf_dataset_test.csv']
    
    # surface_files = ['shape_datasets/ellipse_sdf_surface_dataset_test',
    #                  'shape_datasets/triangle_sdf_surface_dataset_test',
    #                  'shape_datasets/quadrangle_sdf_surface_dataset_test']

    dataset = SdfDataset(dataset_files)
    surface_dataset = SdfDatasetSurface(surface_files)


    # Split dataset into train and test sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # 20% for testing

    # Use random_split to create train and test datasets
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Set seed for reproducibility
    )

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
        batch_size=8,
        shuffle=False,
        num_workers=15,
        collate_fn=collate_fn_surface
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Surface test set size: {len(surface_test_loader)}")

    MAX_EPOCHS = args.max_epochs
    MAX_STEPS = MAX_EPOCHS * len(train_loader)

    # Training setup
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        devices=1,
        logger=TensorBoardLogger(
            name='VAEi', 
            save_dir='./logs', 
            default_hp_metric=False, 
            version=args.run_name
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
            )
        ],
        check_val_every_n_epoch=None,  # Disable validation every epoch
        val_check_interval=4000  # Perform validation every 2000 training steps
    )

    # Initialize model with L1 regularization
    vae_model = VAE(
        input_dim=dataset.feature_dim, 
        latent_dim=3, 
        hidden_dim=128, 
        regularization='l2',   # Use 'l1', 'l2', or None
        reg_weight=1e-4        # Adjust the weight as needed
    )

    # Initialize the trainer
    vae_trainer = LitSdfVAE(
        vae_model=vae_model, 
        learning_rate=1e-4, 
        reg_weight=1e-4, 
        regularization='l2',    # Should match the VAE model's regularization
        warmup_steps=1000, 
        max_steps=MAX_STEPS
    )

    # Train the model
    trainer.fit(vae_trainer, train_loader, val_dataloaders=[test_loader, surface_test_loader])

    # Save model weights
    checkpoint_path = f'model_weights/{args.run_name}.ckpt'
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model weights saved to {checkpoint_path}")

    # Save just the model weights
    model_weights_path = f'model_weights/{args.run_name}.pt'
    torch.save(vae_model.state_dict(), model_weights_path)
    print(f"Model weights saved to {model_weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('--max_epochs', type=int, default=2, help='Maximum number of epochs for training')
    parser.add_argument('--run_name', type=str, default='run_test', help='Name of the run')
    args = parser.parse_args()
    main(args)
