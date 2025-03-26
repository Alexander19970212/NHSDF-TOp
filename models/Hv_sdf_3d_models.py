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

from sklearn.feature_selection import mutual_info_regression
from dataset_generation_new.utils_generation import plot_sdf_heav_item_by_tensor

import os

def compute_closeness(sdf_pred, sdf_target):
    # Mean Absolute Error
    mae = F.l1_loss(sdf_pred, sdf_target.unsqueeze(1), reduction='mean')
    # Root Mean Square Error
    rmse = torch.sqrt(F.mse_loss(sdf_pred, sdf_target.unsqueeze(1), reduction='mean'))
    return mae, rmse

def finite_difference_smoothness_torch(points):
    dx = torch.diff(points, dim=0)
    dy = torch.diff(points, dim=1)
    dx = dx[:, :-1]  # Adjust dx to match the shape of dy
    dy = dy[:-1, :]  # Adjust dy to match the shape of dx
    smoothness = torch.sqrt(dx**2 + dy**2)
    return smoothness.mean()

def finite_difference_smoothness_3d(points: torch.Tensor) -> torch.Tensor:
    """
    Compute the finite difference smoothness metric for a 3D tensor.
    This calculates the finite differences along each of the three spatial dimensions (depth, height, width)
    and returns the average gradient magnitude computed over the overlapping interior region.
    
    Args:
        points (torch.Tensor): A 3D tensor with shape (D, H, W) representing a spatial grid.
    
    Returns:
        torch.Tensor: The mean finite difference smoothness metric.
    """
    # Compute finite differences along each dimension
    dx = torch.diff(points, dim=0)  # shape: (D-1, H, W)
    dy = torch.diff(points, dim=1)  # shape: (D, H-1, W)
    dz = torch.diff(points, dim=2)  # shape: (D, H, W-1)

    # Align the differences to a common interior region:
    # For dx: remove the last row and column to match the region where dy and dz are defined.
    dx_common = dx[:, :-1, :-1]   # shape: (D-1, H-1, W-1)
    # For dy: drop the last element along the depth and width dimensions.
    dy_common = dy[:-1, :, :-1]   # shape: (D-1, H-1, W-1)
    # For dz: drop the last element along the depth and height dimensions.
    dz_common = dz[:-1, :-1, :]   # shape: (D-1, H-1, W-1)

    # Compute the gradient magnitude at each interior point
    gradient_magnitude = torch.sqrt(dx_common**2 + dy_common**2 + dz_common**2)
    
    # Return the mean smoothness metric
    return gradient_magnitude.mean()


class Decoder3D(nn.Module):
    """
    Decoder from DeepSDF:
    https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/networks.py
    """
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder3D, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 2

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 2 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
    
# Define a simple residual block for non-linear deep feature extraction.
class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        # Use a linear projection in the shortcut if dimensions differ.
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return self.act(out + residual) 

class VAE(nn.Module):
    """
    A standard Variational Autoencoder adapted to match the AE_DeepSDF interface for comparison.
    """
    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=32, tau_latent_dim=2,
                 tau_loss_weight=0.1, orthogonality_loss_weight=0.1,
                 kl_weight=1e-4, orthogonality_loss_type=None,
                 regularization=None, reg_weight=1e-4):
        """
        Initializes the Standard VAE.

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
        self.orthogonality_loss_weight = orthogonality_loss_weight
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim

        # Advanced Encoder: Process only shape parameters (exclude coordinate features) using a residual MLP for enhanced 3D feature extraction.
        shape_param_dim = input_dim - 3  # Exclude coordinate features
        
        self.encoder = nn.Sequential(
            nn.Linear(shape_param_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_dim, hidden_dim * 2, hidden_dim * 2),
            ResidualBlock(hidden_dim * 2, hidden_dim, hidden_dim)
        )

        # Mean and log variance layers for latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

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
            nn.Linear(hidden_dim, input_dim-3)
        )

        # Decoder for SDF prediction
        self.decoder_hv_sdf = nn.Sequential(
            nn.Linear(latent_dim+2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
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

        # Initialize mu and logvar layers
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

        # Initialize decoder_input layers
        for layer in self.decoder_input:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize decoder_sdf layers
        for layer in self.decoder_hv_sdf:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def encode(self, x):
        """
        Encodes the input into latent space.

        Args:
            x (Tensor): Input tensor.

        Returns:
            mu (Tensor): Mean of the latent distribution.
            log_var (Tensor): Log variance of the latent distribution.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (Tensor): Mean of the latent distribution.
            log_var (Tensor): Log variance of the latent distribution.

        Returns:
            z (Tensor): Sampled latent vector.
        """
        # std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu      

        # return mu + eps * std

    def forward(self, x, reconstruction=False, Heaviside=True):
        """
        Forward pass through the VAE.

        Args:
            x (Tensor): Input tensor.
            reconstruction (bool, optional): Flag to compute reconstruction. Defaults to False.

        Returns:
            If reconstruction is False:
                recon_x (Tensor): Reconstructed input.
                mu (Tensor): Mean of the latent distribution.
                log_var (Tensor): Log variance of the latent distribution.
            If reconstruction is True:
                recon_x (Tensor): Reconstructed input.
                mu (Tensor): Mean of the latent distribution.
                log_var (Tensor): Log variance of the latent distribution.
                z (Tensor): Sampled latent vector.
        """
        query_points = x[:, :3]
        mu, log_var = self.encode(x[:, 3:])
        z = self.reparameterize(mu, log_var)

        output = {
            "z": z,
            "mu": mu,
            "log_var": log_var
        }

        if Heaviside:
            hv_sdf_decoder_input = torch.cat([z, query_points], dim=1)
            hv_sdf_pred = self.decoder_hv_sdf(hv_sdf_decoder_input)
            output["hv_sdf_pred"] = hv_sdf_pred

        if reconstruction:
            x_reconstructed = self.decoder_input(z)
            output["x_reconstructed"] = x_reconstructed

        return output

    def loss_function(self, loss_args, Heaviside=True, Reconstruction=False):
        """
        Calculates the loss function with optional L1 or L2 regularization.

        Args:
            loss_args (dict): Dictionary containing the following keys:
                - "tau_pred" (Tensor): Predicted tau.
                - "mu" (Tensor): Mean of the latent distribution.
                - "log_var" (Tensor): Log variance of the latent distribution.
                - "z" (Tensor): Latent representations.

        Returns:
            total_loss (Tensor): Combined loss.
            splitted_loss (dict): Dictionary containing individual loss components.
        """
        mu = loss_args["mu"]
        log_var = loss_args["log_var"]
        z = loss_args["z"]

        # KL Divergence loss
        kl_divergence = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(1).mean()

        # Regularization loss
        if self.regularization == 'l1':
            reg_loss = torch.mean(torch.abs(z))
        elif self.regularization == 'l2':
            reg_loss = torch.mean(z.pow(2))
        else:
            reg_loss = 0

        total_loss = (
            + self.reg_weight * reg_loss
            + self.kl_weight * kl_divergence
        )

        splitted_loss = {
            # "total_loss": total_loss,
            "kl_divergence": kl_divergence,
            "reg_loss": reg_loss
        }

        if Heaviside:
            hv_sdf_pred = loss_args["hv_sdf_pred"]
            hv_sdf_target = loss_args["hv_sdf_target"]
            hv_sdf_loss = F.mse_loss(hv_sdf_pred, hv_sdf_target.unsqueeze(1), reduction='mean')
            splitted_loss["hv_sdf_loss"] = hv_sdf_loss
            total_loss += hv_sdf_loss

        if Reconstruction:
            x_reconstructed = loss_args["x_reconstructed"]
            x_original = loss_args["x_original"]
            reconstruction_loss = F.mse_loss(x_reconstructed, x_original, reduction='mean')
            splitted_loss["reconstruction_loss"] = reconstruction_loss
            total_loss += reconstruction_loss

        splitted_loss["total_loss"] = total_loss

        return total_loss, splitted_loss
    
    def reconstruction_loss(self, x_reconstructed, x):
        x_original = x[:, 3:]
        reconstruction_loss = F.mse_loss(x_reconstructed, x_original, reduction='mean')
        return reconstruction_loss, {"reconstruction_loss": reconstruction_loss}

    def hv_sdf(self, z, query_points):
        z_repeated = z.repeat(query_points.shape[0], 1)

        # Decode
        hv_sdf_decoder_input = torch.cat([z_repeated, query_points], dim=1)
        hv_sdf_pred = self.decoder_hv_sdf(hv_sdf_decoder_input)

        return hv_sdf_pred


class VAE_DeepSDF3D(VAE):
    def __init__(self,
                 input_dim: int = 4,
                 latent_dim: int = 2,
                 hidden_dim: int = 32,
                 orthogonality_loss_weight: float = 0.1,
                 kl_weight: float = 1e-4,
                 orthogonality_loss_type=None,
                 regularization=None,
                 reg_weight: float = 1e-4) -> None:
        super().__init__(input_dim=input_dim,
                         latent_dim=latent_dim,
                         hidden_dim=hidden_dim,
                         orthogonality_loss_weight=orthogonality_loss_weight,
                         kl_weight=kl_weight,
                         orthogonality_loss_type=orthogonality_loss_type,
                         regularization=regularization,
                         reg_weight=reg_weight)
        
        network_specs = {
            "latent_size": latent_dim,
            "dims": [512, 512, 512, 512, 512, 512, 512, 512],
            "dropout": list(range(8)),
            "dropout_prob": 0.2,
            "norm_layers": list(range(8)),
            "latent_in": [4],
            "xyz_in_all": False,
            "use_tanh": False,
            "latent_dropout": False,
            "weight_norm": False
        }
        
        self.decoder_hv_sdf = Decoder3D(**network_specs)

    def init_weights(self):
        # Initialize encoder layers
        # Initialize encoder layers
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize mu and logvar layers
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

        # Initialize decoder_input layers
        for layer in self.decoder_input:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

def extract_slices(heaviside_sdf, point_per_side, z_levels):
    z_slice_heaviside_sdf = heaviside_sdf[:-(point_per_side**2)*2]
    xz_slice_heaviside_sdf = heaviside_sdf[-(point_per_side**2)*2:-(point_per_side**2)]
    yz_slice_heaviside_sdf = heaviside_sdf[-(point_per_side**2):]

    ###
    # heaviside_sdf_tensor = torch.from_numpy(z_slice_heaviside_sdf)
    heaviside_sdf_tensor = z_slice_heaviside_sdf.reshape(point_per_side, point_per_side, z_levels)
    heaviside_sdf_tensor = heaviside_sdf_tensor.permute(2, 0, 1)

    heaviside_sdf_tensor = heaviside_sdf_tensor.reshape(heaviside_sdf_tensor.shape[0], -1)
    ###

    # heaviside_sdf_tensor_xz = torch.from_numpy(xz_slice_heaviside_sdf)
    heaviside_sdf_tensor_xz = xz_slice_heaviside_sdf[None, :]
    ###

    # heaviside_sdf_tensor_yz = torch.from_numpy(yz_slice_heaviside_sdf)
    heaviside_sdf_tensor_yz = yz_slice_heaviside_sdf[None, :]

    heaviside_sdf_tensor = torch.cat([heaviside_sdf_tensor, heaviside_sdf_tensor_xz, heaviside_sdf_tensor_yz], dim=0)

    return heaviside_sdf_tensor
        
class Lit3DHvDecoderGlobal(L.LightningModule):
    def __init__(self, vae_model, learning_rate=1e-4, reg_weight=1e-4, 
                 regularization=None, warmup_steps=1000, max_steps=10000,
                 slice_chicking=False, reconstract_dir=None):
        super().__init__()
        self.vae = vae_model
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.regularization = regularization
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.freezing_weights()
        self.save_hyperparameters(logger=False)
        self.slice_chicking = slice_chicking
        

        if self.slice_chicking:
            self.reconstract_dir = reconstract_dir
            self.point_per_side = 25
            self.z_levels = 6
            self.slice_points()

    def freezing_weights(self):
        for param in self.vae.parameters():
            param.requires_grad = True

        for param in self.vae.decoder_input.parameters():
            param.requires_grad = False

        stat = self.count_parameters()

    def count_parameters(self):
        """Count trainable and frozen parameters."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        for name, param in self.vae.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            else:
                frozen_params += num_params

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params
        }

    def forward(self, x):
        return self.vae(x)
   
    def slice_points(self):
        x = np.linspace(-1, 1, self.point_per_side)
        y = np.linspace(-1, 1, self.point_per_side) 
        z = np.linspace(-1, 1, self.z_levels)

        # Create meshgrid
        X, Y, Z = np.meshgrid(x, y, z)

        # Reshape to get array of 3D points
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T


        # Generate points for a slice along the x-z plane (setting y = 0) with a grid of size (point_per_side x point_per_side)
        z_xz = np.linspace(-1, 1, self.point_per_side)  # Create an array for the z-axis with point_per_side samples
        X_xz, Z_xz = np.meshgrid(x, z_xz)            # Use the same x array as before and new z_xz for the grid
        Y_xz = np.zeros_like(X_xz)                    # Fix the y-coordinate at 0 for the x-z plane slice
        points_xz = np.vstack([X_xz.ravel(), Y_xz.ravel(), Z_xz.ravel()]).T
        points_yz = np.vstack([Y_xz.ravel(), Z_xz.ravel(), X_xz.ravel()]).T

        # print(points_xz.shape)
        self.points_to_compute_heaviside = np.vstack([points, points_xz, points_yz])

        z_slice_points = self.points_to_compute_heaviside[:-(self.point_per_side**2)*2]
        xz_slice_points = self.points_to_compute_heaviside[-(self.point_per_side**2)*2:-(self.point_per_side**2)]
        yz_slice_points = self.points_to_compute_heaviside[-(self.point_per_side**2):]

        ###
        points_tensor = torch.from_numpy(z_slice_points).reshape(self.point_per_side, self.point_per_side, self.z_levels, -1)
        points_tensor = points_tensor.permute(2, 0, 1, 3)
        points_tensor = points_tensor.reshape(points_tensor.shape[0], -1, points_tensor.shape[-1])
        # Retain only the first two coordinates (x, y)
        points_tensor = points_tensor[:, :, :2]

        ###
        points_tensor_xz = torch.from_numpy(xz_slice_points).reshape(self.point_per_side, self.point_per_side, -1)
        points_tensor_xz = points_tensor_xz.reshape(1, -1, points_tensor_xz.shape[-1])[:, :, [0, 2]]

        ###
        points_tensor_yz = torch.from_numpy(yz_slice_points).reshape(self.point_per_side, self.point_per_side, -1)
        points_tensor_yz = points_tensor_yz.reshape(1, -1, points_tensor_yz.shape[-1])[:, :, [2, 1]]
        ###

        self.points_tensor = torch.cat([points_tensor, points_tensor_xz, points_tensor_yz], dim=0)
        self.points_to_compute_heaviside = torch.from_numpy(self.points_to_compute_heaviside)

    def prepare_slice_x(self, x_init_list, max_radius_limit=3):

        self.x_slice_list = []

        self.vertices_list = []
        self.arc_radii_list = []

        v1 = np.array([-0.5, -0.5])
        v2 = np.array([0.5, -0.5])

        for i in range(len(x_init_list)):
            item = x_init_list[i]

            # print(item.shape)
            # print(self.points_to_compute_heaviside.shape)

            x_slice_item = item.repeat(self.points_to_compute_heaviside.shape[0], 1)
            x_slice_item[:, :3] = self.points_to_compute_heaviside
            self.x_slice_list.append(x_slice_item)

            v3 = np.array([item[3], item[4]])
            v4 = np.array([item[5], item[6]])

            arc_radii = np.array([item[7], item[8], item[9], item[10]])*max_radius_limit

            self.vertices_list.append(np.array([v1, v2, v3, v4]))
            self.arc_radii_list.append(arc_radii)


    def training_step(self, batch, batch_idx):

        x, sdf, _ = batch

        output = self.vae(x)

        loss_args = output
        loss_args["hv_sdf_target"] = sdf
        loss_args["x_original"] = x
        total_loss, splitted_loss = self.vae.loss_function(loss_args)

        for key, value in splitted_loss.items():
            self.log(f'train_{key}', value, prog_bar=True, batch_size=x.shape[0])

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        if dataloader_idx == 0:
            x, sdf, _ = batch
            output = self.vae(x)
            loss_args = output
            loss_args["hv_sdf_target"] = sdf
            loss_args["x_original"] = x
            total_loss, splitted_loss = self.vae.loss_function(loss_args)

            for key, value in splitted_loss.items():
                self.log(f'val_{key}', value, prog_bar=True, batch_size=x.shape[0])

        if dataloader_idx == 1:

            if self.slice_chicking and batch_idx == 0:
                print(f'Slice chicking at step {self.global_step}')
                heaviside_sdf_tensor_list = []
                for x_init in self.x_slice_list:
                    hv_sdf_pred = self.vae(x_init.to(self.device))["hv_sdf_pred"].squeeze()
                    hv_sdf_pred_tensor = extract_slices(hv_sdf_pred, self.point_per_side, self.z_levels)
                    heaviside_sdf_tensor_list.append(hv_sdf_pred_tensor.detach().cpu())

                filename = os.path.join(self.reconstract_dir, f'{self.global_step}.png')

                plot_sdf_heav_item_by_tensor(
                    self.vertices_list,
                    self.arc_radii_list,
                    heaviside_sdf_tensor_list, #S x N x WH
                    self.points_tensor, # N x WH x 2
                    filename=filename
                )

            x_, sdf_, _ = batch            
            x = x_[0] # batch size 1
            sdf = sdf_[0]
            # print(x.shape, sdf.shape)
            output = self.vae(x)
            hv_sdf_pred = output["hv_sdf_pred"]
            hv_sdf_pred = hv_sdf_pred.squeeze()
            hv_sdf_target = sdf.squeeze()

            hv_sdf_diff = hv_sdf_pred - hv_sdf_target

            # Reshape hv_sdf_diff into a 3D tensor corresponding to the spatial coordinates.
            num_points = hv_sdf_diff.numel()
            grid_size = int(round(num_points ** (1/3)))
            if grid_size ** 3 != num_points:
                raise ValueError(f"hv_sdf_diff with {num_points} elements cannot be reshaped into a 3D cube. Check grid density.")
            
            hv_sdf_diff_3d = hv_sdf_diff.reshape(grid_size, grid_size, grid_size)
            # Optionally log a summary statistic (e.g., mean value) of the reshaped tensor for monitoring
            smoothness = finite_difference_smoothness_3d(hv_sdf_diff_3d)
            self.log("val_smoothness", smoothness, prog_bar=True, batch_size=x.shape[0])


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        warmup_lr_lambda = lambda step: ((step + 1)%self.max_steps) / self.warmup_steps if step < self.warmup_steps else 1
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(self.max_steps - self.warmup_steps), eta_min=0)

        # Start of Selection
        scheduler = {
            'scheduler': SequentialLR(
                optimizer,
                schedulers=[
                    warmup_scheduler,
                    cosine_scheduler,
                    # warmup_scheduler,
                    # cosine_scheduler
                ],
                milestones=[
                    self.warmup_steps,
                    # self.max_steps,
                    # self.max_steps + self.warmup_steps
                ]
            ),
            'interval': 'step',
            'frequency': 1
        }

        return {
            "optimizer": optimizer,
            'lr_scheduler': scheduler
        }

    
