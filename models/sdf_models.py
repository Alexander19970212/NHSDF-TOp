import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import lightning as L
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


class VAE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=32, 
                 regularization=None, reg_weight=1e-4):
        """
        VAE model with optional L1 or L2 regularization.

        Args:
            input_dim (int): Dimension of input features.
            latent_dim (int): Dimension of latent space.
            hidden_dim (int): Dimension of hidden layers.
            regularization (str, optional): Type of regularization ('l1', 'l2', or None).
            reg_weight (float, optional): Weight of the regularization term.
        """
        super(VAE, self).__init__()

        self.regularization = regularization
        self.reg_weight = reg_weight

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim-2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Mean and variance layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder for input reconstruction
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim-2)
        )

        # Decoder for SDF prediction
        self.decoder_sdf = nn.Sequential(
            nn.Linear(latent_dim+2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

        self.init_weights()

    def init_weights(self):
        # Initialize encoder layers
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize mu and var layers
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_var.weight)
        nn.init.zeros_(self.fc_var.bias)

        # Initialize decoder_input layers
        for layer in self.decoder_input:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize decoder_sdf layers
        for layer in self.decoder_sdf:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        query_points = x[:, :2]
        x_encoded = self.encoder(x[:, 2:])  # Only use x[2:] features
        mu = self.fc_mu(x_encoded)
        log_var = self.fc_var(x_encoded)

        # Reparameterization
        z = self.reparameterize(mu, log_var)

        # Decode
        x_reconstructed = self.decoder_input(z)
        sdf_decoder_input = torch.cat([z, query_points], dim=1)
        sdf_pred = self.decoder_sdf(sdf_decoder_input)

        return x_reconstructed, sdf_pred, z  # Return z for regularization

    def loss_function(self, x_reconstructed, x_original, sdf_pred, sdf_target, z, only_recon=False):
        """
        Calculate the loss function with optional L1 or L2 regularization.

        Args:
            x_reconstructed (Tensor): Reconstructed input.
            x_original (Tensor): Original input.
            sdf_pred (Tensor): Predicted SDF.
            sdf_target (Tensor): Target SDF.
            z (Tensor): Latent representations.

        Returns:
            total_loss (Tensor): Combined loss.
            recon_loss (Tensor): Reconstruction loss.
            sdf_loss (Tensor): SDF prediction loss.
            reg_loss (Tensor or 0): Regularization loss.
        """
        # Reconstruction loss for input features
        recon_loss = F.mse_loss(x_reconstructed, x_original[:, 2:], reduction='mean')

        # SDF prediction loss
        sdf_loss = F.mse_loss(sdf_pred, sdf_target.unsqueeze(1), reduction='mean')

        # Regularization loss
        if self.regularization == 'l1':
            reg_loss = torch.mean(torch.abs(z))
        elif self.regularization == 'l2':
            reg_loss = torch.mean(z.pow(2))
        else:
            reg_loss = 0


        if only_recon==False: 
            # Total loss
            # total_loss = recon_loss + sdf_loss + self.reg_weight * reg_loss
            total_loss = sdf_loss + self.reg_weight * reg_loss

        else:
            total_loss = recon_loss

        splitted_loss = {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "sdf_loss": sdf_loss,
            "reg_loss": reg_loss
        }

        return total_loss, splitted_loss
    
    def sdf(self, z, query_points):
        z_repeated = z.repeat(query_points.shape[0], 1)

        # Decode
        sdf_decoder_input = torch.cat([z_repeated, query_points], dim=1)
        sdf_pred = self.decoder_sdf(sdf_decoder_input)

        return sdf_pred
    
class LitSdfVAE(L.LightningModule):
    def __init__(self, vae_model, learning_rate=1e-4, reg_weight=1e-4, 
                 regularization=None, warmup_steps=1000, max_steps=10000):
        super().__init__()
        self.vae = vae_model
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.regularization = regularization
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.only_recon = False
            
        self.save_hyperparameters(logger=False)

    def freeze_decoder_input(self):
        for param in self.vae.parameters():
            param.requires_grad = False

        for param in self.vae.decoder_input.parameters():
            param.requires_grad = True

        self.only_recon = True

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        x, sdf = batch
        x_reconstructed, sdf_pred, z = self.vae(x)

        total_loss, splitted_loss = self.vae.loss_function(
            x_reconstructed, x, sdf_pred, sdf, z, self.only_recon
        )

        for key, value in splitted_loss.items():
            self.log(f'train_{key}', value, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            x, sdf = batch
            # print(f"x: {x.shape}, type: {x.dtype}")
            x_reconstructed, sdf_pred, z = self.vae(x)

            total_loss, splitted_loss = self.vae.loss_function(
                x_reconstructed, x, sdf_pred, sdf, z, self.only_recon
            )

            for key, value in splitted_loss.items():
                self.log(f'val_{key}', value, prog_bar=True)
        else:
            total_loss = 0
            total_splitted_loss = {}
            xs, sdfs, points_s = batch
            for i in range(len(points_s)):
                x = xs[i]
                sdf = sdfs[i]
                points = points_s[i]
                
                x_repeated = x.repeat(points.size(0), 1)
                x_combined = torch.cat((points, x_repeated), dim=1)
                # print(f"x_combined: {x_combined.shape}, type: {x_combined.dtype}")
                x_reconstructed, sdf_pred, z = self.vae(x_combined)

                loss, splitted_loss = self.vae.loss_function(
                    x_reconstructed, x_combined, sdf_pred, sdf, z, self.only_recon
                )

                total_loss += loss

                for key, value in splitted_loss.items():
                    if key not in total_splitted_loss:
                        total_splitted_loss[key] = value
                    else:
                        total_splitted_loss[key] += value

            avg_splitted_loss = {key: value / len(batch) for key, value in total_splitted_loss.items()}

            for key, value in avg_splitted_loss.items():
                self.log(f'val2_{key}', value, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        warmup_lr_lambda = lambda step: (step + 1) / self.warmup_steps if step < self.warmup_steps else 1
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(self.max_steps - self.warmup_steps), eta_min=0)

        scheduler = {
            'scheduler': SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.warmup_steps]
            ),
            'interval': 'step',
            'frequency': 1
        }

        return {
            "optimizer": optimizer,
            'lr_scheduler': scheduler
        }