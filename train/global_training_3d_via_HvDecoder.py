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

from models.Hv_sdf_3d_models import VAE_DeepSDF3D, Lit3DHvDecoderGlobal

try:
    from SDF_dataset import Dataset3DHeavisideSDF, Dataset3DHeavisideSDFGrid
except ImportError:
    from datasets.SDF_dataset import Dataset3DHeavisideSDF, Dataset3DHeavisideSDFGrid

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


def main(args):

    dataset_dir = args.dataset_dir
    train_index_list_csv = args.train_index_list_csv
    test_index_list_csv = args.test_index_list_csv
    config_name = args.config_name
    configs_dir = args.config_dir
    models_dir = args.model_dir
    
    run_name = args.run_name

    train_dataset = Dataset3DHeavisideSDF(dataset_dir, train_index_list_csv)
    test_dataset = Dataset3DHeavisideSDF(dataset_dir, test_index_list_csv)
    test_dataset_grid = Dataset3DHeavisideSDFGrid(f"{dataset_dir}_grid")


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

    test_loader_grid = torch.utils.data.DataLoader(
        test_dataset_grid, 
        batch_size=1,
        shuffle=False,  # No need to shuffle test data
        num_workers=1
    )
    #################################################

    MAX_EPOCHS = args.max_epochs
    MAX_STEPS = MAX_EPOCHS * len(train_loader)

    # Training setup
    trainer = Trainer(
        max_epochs=MAX_EPOCHS, # the first epoch for training all model, the second one for training rec decoder
        accelerator='auto',
        devices=1,
        logger=TensorBoardLogger(
            name='3d_HvDecGlobal', 
            save_dir='./logs', 
            default_hp_metric=False, 
            version=run_name
        ),
        callbacks=[
            callbacks.ModelCheckpoint(
                monitor='val_total_loss',
                mode='min',
                save_top_k=1,
                filename='best-model-{epoch:02d}-{val_total_loss:.2f}'
            ),
            callbacks.EarlyStopping(
                monitor='val_total_loss',
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
    vae_model = VAE_DeepSDF3D(**model_params)

    # Initialize VAE trainer
    trainer_params = config['trainer']['params']
    trainer_params['vae_model'] = vae_model
    trainer_params['max_steps'] = MAX_STEPS
    # vae_trainer = trainers[config['trainer']['type']](**trainer_params)
    vae_trainer = Lit3DHvDecoderGlobal(**trainer_params)

    # Train the model
    # trainer.validate(vae_trainer, dataloaders=[test_loader, test_loader_grid])
    # trainer.fit(vae_trainer, train_loader, val_dataloaders=[test_loader])
    final_metrics = trainer.validate(vae_trainer, dataloaders=[test_loader_grid])

    # Save model weights
    checkpoint_path = f'{models_dir}/{run_name}_HvDecGlobal.ckpt'
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model weights saved to {checkpoint_path}")

    # Save just the model weights
    model_weights_path = f'{models_dir}/{run_name}_HvDecGlobal.pt'
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
    parser.add_argument('--run_name', type=str, default='3d_HvDecGlobal', help='Name of the run')
    parser.add_argument('--model_dir', type=str, default='model_weights', help='Path to the model directory')
    parser.add_argument('--dataset_dir', type=str, default='shape_datasets', help='Path to the dataset')
    parser.add_argument('--train_index_list_csv', type=str, default='train_index_list.csv', help='Path to the train index list')
    parser.add_argument('--test_index_list_csv', type=str, default='test_index_list.csv', help='Path to the test index list')
    parser.add_argument('--config_dir', type=str, default='configs/NN_sdf_experiments/architectures', help='Path to the config directory')
    parser.add_argument('--config_name', type=str, default='AE_DeepSDF', help='Name of the config')
    parser.add_argument('--metrics_file', type=str, default='src/metrics.json', help='Path to the metrics file')
    args = parser.parse_args()
    main(args)
