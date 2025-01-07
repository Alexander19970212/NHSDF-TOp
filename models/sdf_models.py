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

def orthogonality_metric(z, tau_latent_dim):
    z_tau = z[:, :tau_latent_dim]
    z_original = z[:, tau_latent_dim:]
    
    Q_tau, R_tau = torch.linalg.qr(z_tau)
    Q_original, R_original = torch.linalg.qr(z_original)

    orthogonality_loss = torch.linalg.norm(Q_tau.T @ Q_original, ord=2)
    return orthogonality_loss

def orthogonality_metrics(z, target, tau_latent_dim):
    z_tau = z[:, :tau_latent_dim]
    z_original = z[:, tau_latent_dim:]
    z_original_std = torch.std(z_original, dim=0)
    z_tau_std = torch.std(z_tau, dim=0)

    mi_original = mutual_info_regression(z_original.detach().cpu().numpy(), target.detach().cpu().numpy())
    mi_tau = mutual_info_regression(z_tau.detach().cpu().numpy(), target.detach().cpu().numpy())

    metrics = {
        'mi_original': mi_original.mean(),
        'mi_tau': mi_tau.mean(),
        'z_original_std': z_original_std.mean(),
        'z_tau_std': z_tau_std.mean(),
        'z_std_ratio': z_original_std.mean() / z_tau_std.mean(),
        'mi_ratio': mi_original.mean() / mi_tau.mean()
    }

    return metrics

def orth_minQQ_Frobenius(z, tau_latent_dim):
    z_tau = z[:, :tau_latent_dim]
    z_original = z[:, tau_latent_dim:]
    Q_tau, R_tau = torch.linalg.qr(z_tau)
    Q_original, R_original = torch.linalg.qr(z_original)
    return torch.norm(Q_tau.T @ Q_original)

def orth_minWW_Frobenius(z, tau_latent_dim):
    # W, R = torch.linalg.qr(z)
    I = torch.eye(z.shape[1], device=z.device)
    return torch.norm(z.T @ z - I)

def orth_minZZ_Frobenius(z, latent_dim):
    """
    to ensure that the explicit radius dimension remains disentangled from other latent features
    """
    z_radius = z[:, :latent_dim]
    z_original = z[:, latent_dim:]
    orthogonality_loss = torch.norm(z_radius.T @ z_original)
    return orthogonality_loss

orth_losses = {
    'orth_minQQ_Frobenius': orth_minQQ_Frobenius,
    'orth_minZZ_Frobenius': orth_minZZ_Frobenius,
    'orth_minWW_Frobenius': orth_minWW_Frobenius,
    'None': None
}


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 2] + dims + [1]

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
        xyz = input[:, -2:]

        if input.shape[1] > 2 and self.latent_dropout:
            latent_vecs = input[:, :-2]
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
    
class Decoder_loss_old(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Decoder_loss, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
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

    def forward(self, z):
        return self.model(z)


class AE_old(nn.Module):
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
        super(AE, self).__init__()

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

        #decoder for radius sum
        self.decoder_tau = Decoder_loss(latent_dim, hidden_dim)

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
    
    def tau(self, z):
        return self.decoder_tau(z)
    
    def tau_loss(self, tau_pred, tau_target):
        return F.mse_loss(tau_pred, tau_target, reduction='mean')

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
    
class AE_DeepSDF_old(AE_old):
    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=32, 
                 regularization=None, reg_weight=1e-4):
        super(AE_DeepSDF, self).__init__(input_dim, latent_dim, hidden_dim, 
                                        regularization, reg_weight)
        
        NetworkSpecs = {
            "latent_size" : latent_dim,
            "dims" : [ 512, 512, 512, 512, 512, 512, 512, 512 ],
            "dropout" : [0, 1, 2, 3, 4, 5, 6, 7],
            "dropout_prob" : 0.2,
            "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
            "latent_in" : [4],
            "xyz_in_all" : False,
            "use_tanh" : False,
            "latent_dropout" : False,
            "weight_norm" : False
        }
        
        self.decoder_sdf = Decoder(**NetworkSpecs)

class AE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=32, tau_latent_dim=2,
                 tau_loss_weight=0.1, orthogonality_loss_weight=0.1,
                 orthogonality_loss_type=None,
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
        super(AE, self).__init__()

        self.regularization = regularization
        self.reg_weight = reg_weight
        print(f"regularization: {regularization}, reg_weight: {reg_weight}")
        self.tau_latent_dim = tau_latent_dim
        self.tau_loss_weight = tau_loss_weight
        self.orthogonality_loss_weight = orthogonality_loss_weight
        self.latent_dim = latent_dim

        if orthogonality_loss_type is None:
            self.orthogonality_loss = None
        else:
            print(f"Using orthogonality loss: {orthogonality_loss_type}")
            self.orthogonality_loss = orth_losses[orthogonality_loss_type]

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
            nn.Linear(hidden_dim, latent_dim)
        )

        #decoder for radius sum
        self.decoder_tau = nn.Linear(tau_latent_dim, 1)

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

        self.init_weights()

    def init_weights(self):
        # Initialize encoder layers
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize decoder_input layers
        for layer in self.decoder_input:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize decoder_tau layers
        nn.init.xavier_uniform_(self.decoder_tau.weight)
        nn.init.zeros_(self.decoder_tau.bias)

        # Initialize decoder_sdf layers
        for layer in self.decoder_sdf:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, reconstruction=False):
        # Encode
        query_points = x[:, :2]
        z = self.encoder(x[:, 2:])  # Only use x[2:] features

        z_tau = z[:, :self.tau_latent_dim]
        tau_pred = self.decoder_tau(z_tau)

        sdf_decoder_input = torch.cat([z, query_points], dim=1)
        sdf_pred = self.decoder_sdf(sdf_decoder_input)

        output = {
            "tau_pred": tau_pred,
            "sdf_pred": sdf_pred,
            "z": z
        }

        if reconstruction:
            x_reconstructed = self.decoder_input(z)
            output["x_reconstructed"] = x_reconstructed

        return output
    
    def tau(self, z):
        z_radius = z[:, :self.tau_latent_dim]
        return self.decoder_tau(z_radius)
    
    # def orthogonality_loss(self, z):
    #     return torch.norm(z.T @ z - torch.eye(z.shape[1]), p='fro')
        

    def loss_function(self, loss_args):
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
            radius_sum_loss (Tensor): Reconstruction loss.
            sdf_loss (Tensor): SDF prediction loss.
            reg_loss (Tensor or 0): Regularization loss.
        """
        tau_pred = loss_args["tau_pred"]
        tau_target = loss_args["tau_target"]
        sdf_pred = loss_args["sdf_pred"]
        sdf_target = loss_args["sdf_target"]
        z = loss_args["z"]

        # Reconstruction loss for input features
        tau_loss = F.mse_loss(tau_pred.flatten(), tau_target.flatten(), reduction='mean')

        # SDF prediction loss
        sdf_loss = F.mse_loss(sdf_pred, sdf_target.unsqueeze(1), reduction='mean')

        # Orthogonality loss
        if self.orthogonality_loss is not None:
            orthogonality_loss = self.orthogonality_loss(z, self.tau_latent_dim)
        else:
            orthogonality_loss = 0

        # if not self.training:
        #     orthogonality_metric_value = orthogonality_metric(z, self.tau_latent_dim)
        # else:
        #     orthogonality_metric_value = 0

        # Regularization loss
        if self.regularization == 'l1':
            reg_loss = torch.mean(torch.abs(z))
        elif self.regularization == 'l2':
            reg_loss = torch.mean(z.pow(2))
        else:
            reg_loss = 0

        total_loss = (
            sdf_loss
            + self.tau_loss_weight * tau_loss 
            + self.orthogonality_loss_weight * orthogonality_loss 
            + self.reg_weight * reg_loss
        )
       
        splitted_loss = {
            "total_loss": total_loss,
            "tau_loss": tau_loss,
            "sdf_loss": sdf_loss,
            "orthogonality_loss": orthogonality_loss,
            "reg_loss": reg_loss
        }

        # if not self.training:
        #     splitted_loss["orthogonality_metric"] = orthogonality_metric_value

        return total_loss, splitted_loss
    
    def reconstruction_loss(self, x_reconstructed, x):
        x_original = x[:, 2:]
        reconstruction_loss = F.mse_loss(x_reconstructed, x_original, reduction='mean')
        return reconstruction_loss, {"reconstruction_loss": reconstruction_loss}
    
    def sdf(self, z, query_points):
        z_repeated = z.repeat(query_points.shape[0], 1)

        # Decode
        sdf_decoder_input = torch.cat([z_repeated, query_points], dim=1)
        sdf_pred = self.decoder_sdf(sdf_decoder_input)

        return sdf_pred
    
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
        self.tau_latent_dim = tau_latent_dim
        self.tau_loss_weight = tau_loss_weight
        self.orthogonality_loss_weight = orthogonality_loss_weight
        self.kl_weight = kl_weight

        if orthogonality_loss_type is None:
            self.orthogonality_loss = None
        else:
            self.orthogonality_loss = orth_losses[orthogonality_loss_type]

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

        #decoder for radius sum
        self.decoder_tau = nn.Linear(tau_latent_dim, 1)

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
        for layer in self.decoder_sdf:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize decoder_tau layers
        nn.init.xavier_uniform_(self.decoder_tau.weight)
        nn.init.zeros_(self.decoder_tau.bias)

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
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, reconstruction=False):
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
        query_points = x[:, :2]
        mu, log_var = self.encode(x[:, 2:])
        z = self.reparameterize(mu, log_var)

        z_tau = z[:, :self.tau_latent_dim]
        tau_pred = self.decoder_tau(z_tau)

        sdf_decoder_input = torch.cat([z, query_points], dim=1)
        sdf_pred = self.decoder_sdf(sdf_decoder_input)

        output = {
            "tau_pred": tau_pred,
            "sdf_pred": sdf_pred,
            "z": z,
            "mu": mu,
            "log_var": log_var
        }

        if reconstruction:
            x_reconstructed = self.decoder_input(z)
            output["x_reconstructed"] = x_reconstructed

        return output
        
    def tau(self, z):
        z_radius = z[:, :self.tau_latent_dim]
        return self.decoder_tau(z_radius)

    def loss_function(self, loss_args):
        """
        Calculates the loss function with optional L1 or L2 regularization.

        Args:
            loss_args (dict): Dictionary containing the following keys:
                - "recon_x" (Tensor): Reconstructed input.
                - "x_original" (Tensor): Original input.
                - "mu" (Tensor): Mean of the latent distribution.
                - "log_var" (Tensor): Log variance of the latent distribution.
                - "z" (Tensor): Latent representations.
            only_recon (bool, optional): Flag to compute only reconstruction loss. Defaults to False.

        Returns:
            total_loss (Tensor): Combined loss.
            splitted_loss (dict): Dictionary containing individual loss components.
        """
        mu = loss_args["mu"]
        log_var = loss_args["log_var"]
        tau_pred = loss_args["tau_pred"]
        tau_target = loss_args["tau_target"]
        sdf_pred = loss_args["sdf_pred"]
        sdf_target = loss_args["sdf_target"]
        z = loss_args["z"]

        # Reconstruction loss
        # recon_loss = F.mse_loss(recon_x, x_original, reduction='mean')
        tau_loss = F.mse_loss(tau_pred.flatten(), tau_target.flatten(), reduction='mean')

        # SDF loss
        sdf_loss = F.mse_loss(sdf_pred, sdf_target.unsqueeze(1), reduction='mean')

        # Orthogonality loss
        if self.orthogonality_loss is not None:
            orthogonality_loss = self.orthogonality_loss(z, self.tau_latent_dim)
        else:
            orthogonality_loss = 0

        # KL Divergence loss
        kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # print(self.kl_weight)
        # print(f"kl_divergence: {kl_divergence}")

        # if not self.training:
        #     orthogonality_metric_value = orthogonality_metric(z, self.tau_latent_dim)
        # else:
        #     orthogonality_metric_value = 0

        # Regularization loss
        if self.regularization == 'l1':
            reg_loss = torch.mean(torch.abs(z))
        elif self.regularization == 'l2':
            reg_loss = torch.mean(z.pow(2))
        else:
            reg_loss = 0

        total_loss = (
            sdf_loss
            + self.tau_loss_weight * tau_loss 
            + self.orthogonality_loss_weight * orthogonality_loss 
            + self.reg_weight * reg_loss
            + self.kl_weight * kl_divergence
        )

        splitted_loss = {
            "total_loss": total_loss,
            "kl_divergence": kl_divergence,
            "sdf_loss": sdf_loss,
            "tau_loss": tau_loss,
            "orthogonality_loss": orthogonality_loss,
            "reg_loss": reg_loss
        }

        # if not self.training:
        #     splitted_loss["orthogonality_metric"] = orthogonality_metric_value

        return total_loss, splitted_loss
    
    def reconstruction_loss(self, x_reconstructed, x):
        x_original = x[:, 2:]
        reconstruction_loss = F.mse_loss(x_reconstructed, x_original, reduction='mean')
        return reconstruction_loss, {"reconstruction_loss": reconstruction_loss}

    def sdf(self, z, query_points):
        z_repeated = z.repeat(query_points.shape[0], 1)

        # Decode
        sdf_decoder_input = torch.cat([z_repeated, query_points], dim=1)
        sdf_pred = self.decoder_sdf(sdf_decoder_input)

        return sdf_pred

#############################   MMD   #############################

def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

class MMD_VAE(AE):
    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=32, tau_latent_dim=2, 
                 tau_loss_weight=0.1, orthogonality_loss_weight=0.1,
                 regularization=None,  orthogonality_loss_type=None, reg_weight=1e-4, mmd_weight=1e-4):
        super(MMD_VAE, self).__init__(input_dim, latent_dim, hidden_dim, tau_latent_dim,
                                        tau_loss_weight, orthogonality_loss_weight,
                                        orthogonality_loss_type, regularization, reg_weight)
        
        self.mmd_weight = mmd_weight
        if orthogonality_loss_type is None:
            self.orthogonality_loss = None
        else:
            self.orthogonality_loss = orth_losses[orthogonality_loss_type]
        
    def loss_function(self, loss_args):
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
            radius_sum_loss (Tensor): Reconstruction loss.
            sdf_loss (Tensor): SDF prediction loss.
            reg_loss (Tensor or 0): Regularization loss.
        """
        tau_pred = loss_args["tau_pred"]
        tau_target = loss_args["tau_target"]
        sdf_pred = loss_args["sdf_pred"]
        sdf_target = loss_args["sdf_target"]
        z = loss_args["z"]

        # Compute mmd-loss
        true_samples = torch.randn(200, self.latent_dim, requires_grad=False, device=z.device)
        mmd_loss = MMD(true_samples, z)

        # Reconstruction loss for input features
        tau_loss = F.mse_loss(tau_pred.flatten(), tau_target.flatten(), reduction='mean')

        # SDF prediction loss
        sdf_loss = F.mse_loss(sdf_pred, sdf_target.unsqueeze(1), reduction='mean')

        # Orthogonality loss
        if self.orthogonality_loss is not None:
            orthogonality_loss = self.orthogonality_loss(z, self.tau_latent_dim)
        else:
            orthogonality_loss = 0

        # if not self.training:
        #     orthogonality_metric_value = orthogonality_metric(z, self.tau_latent_dim)
        # else:
        #     orthogonality_metric_value = 0

        # Regularization loss
        if self.regularization == 'l1':
            reg_loss = torch.mean(torch.abs(z))
        elif self.regularization == 'l2':
            reg_loss = torch.mean(z.pow(2))
        else:
            reg_loss = 0

        total_loss = (
            sdf_loss
            + self.tau_loss_weight * tau_loss 
            + self.orthogonality_loss_weight * orthogonality_loss 
            + self.reg_weight * reg_loss
            + self.mmd_weight * mmd_loss
        )

        
        splitted_loss = {
            "total_loss": total_loss,
            "tau_loss": tau_loss,
            "sdf_loss": sdf_loss,
            "orthogonality_loss": orthogonality_loss,
            "reg_loss": reg_loss,
            "mmd_loss": mmd_loss
        }

        # if not self.training:
        #     splitted_loss["orthogonality_metric"] = orthogonality_metric_value

        return total_loss, splitted_loss


class AE_DeepSDF(AE):
    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=32, tau_latent_dim=2, 
                 tau_loss_weight=0.1, orthogonality_loss_weight=0.1, orthogonality_loss_type=None,
                 regularization=None, reg_weight=1e-4):
        super(AE_DeepSDF, self).__init__(input_dim, latent_dim, hidden_dim, tau_latent_dim,
                                        tau_loss_weight, orthogonality_loss_weight, orthogonality_loss_type,
                                        regularization, reg_weight)
        
        NetworkSpecs = {
            "latent_size" : latent_dim,
            "dims" : [ 512, 512, 512, 512, 512, 512, 512, 512 ],
            "dropout" : [0, 1, 2, 3, 4, 5, 6, 7],
            "dropout_prob" : 0.2,
            "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
            "latent_in" : [4],
            "xyz_in_all" : False,
            "use_tanh" : False,
            "latent_dropout" : False,
            "weight_norm" : False
        }
        
        self.decoder_sdf = Decoder(**NetworkSpecs)

    def init_weights(self):
        # Initialize encoder layers
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize decoder_input layers
        for layer in self.decoder_input:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize decoder_tau layers
        nn.init.xavier_uniform_(self.decoder_tau.weight)
        nn.init.zeros_(self.decoder_tau.bias)

class VAE_DeepSDF(VAE):
    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=32, tau_latent_dim=2, 
                 tau_loss_weight=0.1, orthogonality_loss_weight=0.1,
                 kl_weight=1e-4, orthogonality_loss_type = None,
                 regularization=None, reg_weight=1e-4):
        super(VAE_DeepSDF, self).__init__(input_dim, latent_dim, hidden_dim, tau_latent_dim,
                                        tau_loss_weight, orthogonality_loss_weight, kl_weight,
                                        orthogonality_loss_type, regularization, reg_weight)
        
        NetworkSpecs = {
            "latent_size" : latent_dim,
            "dims" : [ 512, 512, 512, 512, 512, 512, 512, 512 ],
            "dropout" : [0, 1, 2, 3, 4, 5, 6, 7],
            "dropout_prob" : 0.2,
            "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
            "latent_in" : [4],
            "xyz_in_all" : False,
            "use_tanh" : False,
            "latent_dropout" : False,
            "weight_norm" : False
        }
        
        self.decoder_sdf = Decoder(**NetworkSpecs)

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

        # Initialize decoder_tau layers
        nn.init.xavier_uniform_(self.decoder_tau.weight)
        nn.init.zeros_(self.decoder_tau.bias)

class MMD_VAE_DeepSDF(MMD_VAE):
    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=32, tau_latent_dim=2, 
                 tau_loss_weight=0.1, orthogonality_loss_weight=0.1,
                 regularization=None, orthogonality_loss_type=None, reg_weight=1e-4, mmd_weight=1e-4):
        super(MMD_VAE_DeepSDF, self).__init__(input_dim, latent_dim, hidden_dim, tau_latent_dim,
                                        tau_loss_weight, orthogonality_loss_weight,
                                        regularization, orthogonality_loss_type, reg_weight, mmd_weight)
        
        NetworkSpecs = {
            "latent_size" : latent_dim,
            "dims" : [ 512, 512, 512, 512, 512, 512, 512, 512 ],
            "dropout" : [0, 1, 2, 3, 4, 5, 6, 7],
            "dropout_prob" : 0.2,
            "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
            "latent_in" : [4],
            "xyz_in_all" : False,
            "use_tanh" : False,
            "latent_dropout" : False,
            "weight_norm" : False
        }
        
        self.decoder_sdf = Decoder(**NetworkSpecs)

    def init_weights(self):
        # Initialize encoder layers
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize decoder_input layers
        for layer in self.decoder_input:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize decoder_tau layers
        nn.init.xavier_uniform_(self.decoder_tau.weight)
        nn.init.zeros_(self.decoder_tau.bias)

        
class LitSdfAE(L.LightningModule):
    def __init__(self, vae_model, learning_rate=1e-4, reg_weight=1e-4, 
                 regularization=None, warmup_steps=1000, max_steps=10000):
        super().__init__()
        self.vae = vae_model
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.regularization = regularization
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.reconstruction_decoder_training = False
            
        self.save_hyperparameters(logger=False)

    def freezing_weights(self, rec_decoder_training=False):
        for param in self.vae.parameters():
            param.requires_grad = not rec_decoder_training

        for param in self.vae.decoder_input.parameters():
            param.requires_grad = rec_decoder_training

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):

        epoch = self.current_epoch
        dataloader_idx = epoch % 2

        x, sdf, tau = batch[dataloader_idx]
        
        if dataloader_idx == 0:
            if self.reconstruction_decoder_training==True:
                self.freezing_weights(rec_decoder_training=False)
                self.reconstruction_decoder_training = False
            output = self.vae(x)
            loss_args = output
            loss_args["tau_target"] = tau
            loss_args["sdf_target"] = sdf
            total_loss, splitted_loss = self.vae.loss_function(loss_args)
        else:
            if self.reconstruction_decoder_training==False:
                self.freezing_weights(rec_decoder_training=True)
                self.reconstruction_decoder_training = True
            output  = self.vae(x, reconstruction=True)
            total_loss, splitted_loss = self.vae.reconstruction_loss(output["x_reconstructed"], x)

        for key, value in splitted_loss.items():
            self.log(f'train_{key}', value, prog_bar=True, batch_size=x.shape[0])

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            x, sdf, tau = batch
            # print(f"x: {x.shape}, type: {x.dtype}")
            output = self.vae(x, reconstruction=True)
            loss_args = output
            loss_args["tau_target"] = tau
            loss_args["sdf_target"] = sdf

            total_loss, splitted_loss = self.vae.loss_function(loss_args)

            for key, value in splitted_loss.items():
                self.log(f'val_{key}', value, prog_bar=True)

            total_loss, splitted_loss = self.vae.reconstruction_loss(output["x_reconstructed"], x)            
            for key, value in splitted_loss.items():
                self.log(f'val_reconstruction_{key}', value, prog_bar=True)

        elif dataloader_idx == 1:
            # total_loss = 0
            # total_splitted_loss = {}
            xs, sdfs, points_s = batch
            total_mae = 0
            total_rmse = 0
            total_smoothness_pred = 0
            total_smoothness_diff = 0
            total_diff_smoothness = 0

            for i in range(len(points_s)):
                x = xs[i]
                sdf = sdfs[i]
                points = points_s[i]
                # radius_sum = radius_sums[i]

                # Attention: it is supposed that points are from a full grid
                points_per_side = int(np.sqrt(points.shape[0]))
                
                x_repeated = x.repeat(points.size(0), 1)
                x_combined = torch.cat((points, x_repeated), dim=1)
                output = self.vae(x_combined)
                tau_pred = output["tau_pred"]
                sdf_pred = output["sdf_pred"]
                # z = output["z"]

                sdf_diff = sdf_pred.squeeze() - sdf.squeeze()

                loss_args = output
                # loss_args["tau_target"] = tau
                # loss_args["sdf_target"] = sdf
                # loss, splitted_loss = self.vae.loss_function(loss_args)

                # total_loss += loss

                # for key, value in splitted_loss.items():
                #     if key not in total_splitted_loss:
                #         total_splitted_loss[key] = value
                #     else:
                #         total_splitted_loss[key] += value

                # Compute metrics
                mae, rmse = compute_closeness(sdf_pred, sdf)
                sdf_pred_reshaped = sdf_pred.reshape(points_per_side, points_per_side)
                sdf_target_reshaped = sdf.reshape(points_per_side, points_per_side)
                sdf_diff_reshaped = sdf_diff.reshape(points_per_side, points_per_side)
                smoothness_pred = finite_difference_smoothness_torch(sdf_pred_reshaped)
                smoothness_target = finite_difference_smoothness_torch(sdf_target_reshaped)
                smoothness_diff = finite_difference_smoothness_torch(sdf_diff_reshaped)
                diff_smoothness = smoothness_pred - smoothness_target

                total_mae += mae
                total_rmse += rmse
                total_smoothness_pred += smoothness_pred
                total_smoothness_diff += smoothness_diff
                total_diff_smoothness += diff_smoothness

            # avg_splitted_loss = {key: value / len(batch) for key, value in total_splitted_loss.items()}
            avg_mae = total_mae / len(points_s)
            avg_rmse = total_rmse / len(points_s)
            avg_smoothness_pred = total_smoothness_pred / len(points_s)
            avg_smoothness_diff = total_smoothness_diff / len(points_s)
            avg_diff_smoothness = total_diff_smoothness / len(points_s)

            # for key, value in avg_splitted_loss.items():
            #     self.log(f'val_{key}', value, prog_bar=True)

            self.log('val_mae', avg_mae, prog_bar=True)
            self.log('val_rmse', avg_rmse, prog_bar=True)
            self.log('val_smoothness_pred', avg_smoothness_pred, prog_bar=True)
            self.log('val_smoothness_diff', avg_smoothness_diff, prog_bar=True)
            self.log('val_diff_smoothness', avg_diff_smoothness, prog_bar=True)

        if dataloader_idx == 2:
            x, sdf, tau = batch
            # print(f"x: {x.shape}, type: {x.dtype}")

            total_metrics = {}
            total_tau_loss = 0

            for i in range(len(x)):
                x_i = x[i]
                # sdf_i = sdf[i]
                tau_i = tau[i]
                output = self.vae(x_i)
                tau_pred = output["tau_pred"]
                sdf_pred = output["sdf_pred"]
                z = output["z"]       

                # tau_loss = self.vae.radius_sum_loss(tau_pred, tau_i)
                tau_loss = F.mse_loss(tau_pred.flatten(), tau_i.flatten(), reduction='mean')
                splitted_metrics = orthogonality_metrics(z, tau_i, self.vae.tau_latent_dim)
                total_tau_loss += tau_loss
                for key, value in splitted_metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = value
                    else:
                        total_metrics[key] += value

            for key, value in total_metrics.items():
                self.log(f'val_{key}', value / len(x), prog_bar=True)

            self.log('val_tau_loss', total_tau_loss / len(x), prog_bar=True)

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
                    warmup_scheduler,
                    cosine_scheduler
                ],
                milestones=[
                    self.warmup_steps,
                    self.max_steps,
                    self.max_steps + self.warmup_steps
                ]
            ),
            'interval': 'step',
            'frequency': 1
        }

        return {
            "optimizer": optimizer,
            'lr_scheduler': scheduler
        }

################################################################################################
# code partially from https://github.com/AliLotfi92/InfoMaxVAE.git

def permute_dims(z):
    """
    function to permute z based on indicies
    """
    assert z.dim() == 2
    B, _ = z.size()
    perm = torch.randperm(B)
    perm_z = z[perm]
    return perm_z

class MINE_Critic(nn.Module):
    def __init__(self, latent_dim, feature_dim, hidden_size=128):
        super(MINE_Critic, self).__init__()
        self.fc1 = nn.Linear(latent_dim + feature_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, z, y):
        # Concatenate latent vectors and features
        # print(f"z: {z.shape}, y: {y.shape}")
        joint = torch.cat([z, y[:, None]], dim=1)
        out = F.relu(self.fc1(joint))
        out = F.relu(self.fc2(out))
        score = self.fc3(out)
        return score
    
class LitSdfAE_MINE(L.LightningModule):
    def __init__(self, vae_model, learning_rate=1e-4, reg_weight=1e-4,
                 info_weight=1e-2, regularization=None, warmup_steps=1000, max_steps=10000):
        super().__init__()
        self.vae = vae_model
        self.mine = MINE_Critic(self.vae.latent_dim - self.vae.tau_latent_dim, 1)
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.regularization = regularization
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.reconstruction_decoder_training = False

        self.info_weight = info_weight
            
        self.save_hyperparameters(logger=True)
        self.automatic_optimization = False

    def freezing_weights(self, rec_decoder_training=False):
        for param in self.vae.parameters():
            param.requires_grad = not rec_decoder_training

        for param in self.vae.decoder_input.parameters():
            param.requires_grad = rec_decoder_training

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):

        vae_optimizer, mine_optimizer = self.optimizers()

        epoch = self.current_epoch
        dataloader_idx = epoch % 2

        x, sdf, tau = batch[dataloader_idx]
        
        if dataloader_idx == 0:
            if self.reconstruction_decoder_training==True:
                self.freezing_weights(rec_decoder_training=False)
                self.reconstruction_decoder_training = False

            ### VAE training ###
            output = self.vae(x)
            loss_args = output
            loss_args["tau_target"] = tau
            loss_args["sdf_target"] = sdf
            total_loss, splitted_loss = self.vae.loss_function(loss_args)

            info_xz = self.estimate_mi(output["z"][:, self.vae.tau_latent_dim:], tau)
            splitted_loss["info_xz"] = info_xz
            total_loss -= self.info_weight * info_xz

            vae_optimizer.zero_grad()
            # self.manual_backward(total_loss)
            total_loss.backward(retain_graph=True)
            vae_optimizer.step()

            mine_optimizer.zero_grad()
            info_xz = self.estimate_mi(output["z"][:, self.vae.tau_latent_dim:], tau)
            # self.manual_backward(info_xz)
            info_xz.backward(inputs=list(self.mine.parameters()))
            mine_optimizer.step()

        else:
            if self.reconstruction_decoder_training==False:
                self.freezing_weights(rec_decoder_training=True)
                self.reconstruction_decoder_training = True
            output  = self.vae(x, reconstruction=True)
            total_loss, splitted_loss = self.vae.reconstruction_loss(output["x_reconstructed"], x)

            vae_optimizer.zero_grad()
            # self.manual_backward(total_loss)
            total_loss.backward(retain_graph=True)
            vae_optimizer.step()

        for key, value in splitted_loss.items():
            self.log(f'train_{key}', value, prog_bar=True, batch_size=x.shape[0])

        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            x, sdf, tau = batch
            # print(f"x: {x.shape}, type: {x.dtype}")
            output = self.vae(x, reconstruction=True)
            loss_args = output
            loss_args["tau_target"] = tau
            loss_args["sdf_target"] = sdf

            total_loss, splitted_loss = self.vae.loss_function(loss_args)

            for key, value in splitted_loss.items():
                self.log(f'val_{key}', value, prog_bar=True)

            total_loss, splitted_loss = self.vae.reconstruction_loss(output["x_reconstructed"], x)            
            for key, value in splitted_loss.items():
                self.log(f'val_reconstruction_{key}', value, prog_bar=True)

        elif dataloader_idx == 1:
            # total_loss = 0
            # total_splitted_loss = {}
            xs, sdfs, points_s = batch
            total_mae = 0
            total_rmse = 0
            total_smoothness_pred = 0
            total_smoothness_diff = 0
            total_diff_smoothness = 0

            for i in range(len(points_s)):
                x = xs[i]
                sdf = sdfs[i]
                points = points_s[i]
                # radius_sum = radius_sums[i]

                # Attention: it is supposed that points are from a full grid
                points_per_side = int(np.sqrt(points.shape[0]))
                
                x_repeated = x.repeat(points.size(0), 1)
                x_combined = torch.cat((points, x_repeated), dim=1)
                output = self.vae(x_combined)
                tau_pred = output["tau_pred"]
                sdf_pred = output["sdf_pred"]
                # z = output["z"]

                sdf_diff = sdf_pred.squeeze() - sdf.squeeze()

                loss_args = output
                # loss_args["tau_target"] = tau
                # loss_args["sdf_target"] = sdf
                # loss, splitted_loss = self.vae.loss_function(loss_args)

                # total_loss += loss

                # for key, value in splitted_loss.items():
                #     if key not in total_splitted_loss:
                #         total_splitted_loss[key] = value
                #     else:
                #         total_splitted_loss[key] += value

                # Compute metrics
                mae, rmse = compute_closeness(sdf_pred, sdf)
                sdf_pred_reshaped = sdf_pred.reshape(points_per_side, points_per_side)
                sdf_target_reshaped = sdf.reshape(points_per_side, points_per_side)
                sdf_diff_reshaped = sdf_diff.reshape(points_per_side, points_per_side)
                smoothness_pred = finite_difference_smoothness_torch(sdf_pred_reshaped)
                smoothness_target = finite_difference_smoothness_torch(sdf_target_reshaped)
                smoothness_diff = finite_difference_smoothness_torch(sdf_diff_reshaped)
                diff_smoothness = smoothness_pred - smoothness_target

                total_mae += mae
                total_rmse += rmse
                total_smoothness_pred += smoothness_pred
                total_smoothness_diff += smoothness_diff
                total_diff_smoothness += diff_smoothness

            # avg_splitted_loss = {key: value / len(batch) for key, value in total_splitted_loss.items()}
            avg_mae = total_mae / len(points_s)
            avg_rmse = total_rmse / len(points_s)
            avg_smoothness_pred = total_smoothness_pred / len(points_s)
            avg_smoothness_diff = total_smoothness_diff / len(points_s)
            avg_diff_smoothness = total_diff_smoothness / len(points_s)

            # for key, value in avg_splitted_loss.items():
            #     self.log(f'val_{key}', value, prog_bar=True)

            self.log('val_mae', avg_mae, prog_bar=True)
            self.log('val_rmse', avg_rmse, prog_bar=True)
            self.log('val_smoothness_pred', avg_smoothness_pred, prog_bar=True)
            self.log('val_smoothness_diff', avg_smoothness_diff, prog_bar=True)
            self.log('val_diff_smoothness', avg_diff_smoothness, prog_bar=True)

        if dataloader_idx == 2:
            x, sdf, tau = batch
            # print(f"x: {x.shape}, type: {x.dtype}")

            total_metrics = {}
            total_tau_loss = 0

            for i in range(len(x)):
                x_i = x[i]
                # sdf_i = sdf[i]
                tau_i = tau[i]
                output = self.vae(x_i)
                tau_pred = output["tau_pred"]
                sdf_pred = output["sdf_pred"]
                z = output["z"]       

                # tau_loss = self.vae.radius_sum_loss(tau_pred, tau_i)
                tau_loss = F.mse_loss(tau_pred.flatten(), tau_i.flatten(), reduction='mean')
                splitted_metrics = orthogonality_metrics(z, tau_i, self.vae.tau_latent_dim)
                total_tau_loss += tau_loss
                for key, value in splitted_metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = value
                    else:
                        total_metrics[key] += value

            for key, value in total_metrics.items():
                self.log(f'val_{key}', value / len(x), prog_bar=True)

            self.log('val_tau_loss', total_tau_loss / len(x), prog_bar=True)

    def estimate_mi(self, z, tau):
        # pass x_true and learned features z from the discriminator
        d_xz = self.mine(z, tau)

        z_perm = permute_dims(z)
        d_x_z = self.mine(z_perm, tau)

        # TODO: check if this is correct

        info_xz = -(d_xz.mean() - (torch.exp(d_x_z - 1).mean()))

        return info_xz
    
    def configure_optimizers(self):
        vae_optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.learning_rate)
        mine_optimizer = torch.optim.AdamW(self.mine.parameters(), lr=self.learning_rate)

        warmup_lr_lambda = lambda step: ((step + 1)%self.max_steps) / self.warmup_steps if step < self.warmup_steps else 1
        warmup_scheduler = LambdaLR(vae_optimizer, lr_lambda=warmup_lr_lambda)

        cosine_scheduler = CosineAnnealingLR(vae_optimizer, T_max=(self.max_steps - self.warmup_steps), eta_min=0)

        # Start of Selection
        vae_scheduler = {
            'scheduler': SequentialLR(
                vae_optimizer,
                schedulers=[
                    warmup_scheduler,
                    cosine_scheduler,
                    warmup_scheduler,
                    cosine_scheduler
                ],
                milestones=[
                    self.warmup_steps,
                    self.max_steps,
                    self.max_steps + self.warmup_steps
                ]
            ),
            'interval': 'step',
            'frequency': 1
        }

        mine_scheduler = {
            'scheduler': LambdaLR(
                mine_optimizer,
                lr_lambda=lambda step: 1.0 - (step / self.max_steps)
            ),
            'interval': 'step',
            'frequency': 1
        }

        return [vae_optimizer, mine_optimizer], [vae_scheduler, mine_scheduler]