"""

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rbf_kernel(T, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out) ** 2
    distance_matrix_scaled = distance_matrix / length_scale**2
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def diffusion_kernel(T, length_scale):
    assert length_scale < 0.5, "length_scale has to be smaller than 0.5 for the kernel matrix to be diagonally dominant"
    sigmas = torch.ones(T, T) * length_scale
    sigmas_tridiag = torch.diagonal(sigmas, offset=0, dim1=-2, dim2=-1)
    sigmas_tridiag += torch.diagonal(sigmas, offset=1, dim1=-2, dim2=-1)
    sigmas_tridiag += torch.diagonal(sigmas, offset=-1, dim1=-2, dim2=-1)
    kernel_matrix = sigmas_tridiag + torch.eye(T) * (1.0 - length_scale)
    return kernel_matrix


def matern_kernel(T, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = torch.abs(xs_in - xs_out)
    distance_matrix_scaled = distance_matrix / torch.sqrt(length_scale).type(torch.float32)
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def cauchy_kernel(T, sigma, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out) ** 2
    distance_matrix_scaled = distance_matrix / length_scale**2
    kernel_matrix = sigma / (distance_matrix_scaled + 1.0)

    alpha = 0.001
    eye = torch.eye(kernel_matrix.shape[-1])
    return kernel_matrix + alpha * eye


def make_nn(input_size, output_size, hidden_sizes):
    """This function used to creates fully connected neural network.

    Parameters
    ----------
    input_size : int,
        the dimension of input embeddings

    output_size : int,
        the dimension of out embeddings

    hidden_sizes : tuple,
        the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers

    Returns
    -------
    output: tensor
        the processing embeddings
    """
    layers = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            layers.append(nn.Linear(in_features=input_size, out_features=hidden_sizes[i]))
        else:
            layers.append(nn.Linear(in_features=hidden_sizes[i - 1], out_features=hidden_sizes[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_size))
    return nn.Sequential(*layers)


class CustomConv1d(torch.nn.Conv1d):
    def __init(self, in_channels, out_channels, kernel_size, padding):
        super().__init__(in_channels, out_channels, kernel_size, padding)

    def forward(self, x):
        if len(x.shape) > 2:
            shape = list(np.arange(len(x.shape)))
            new_shape = [0, shape[-1]] + shape[1:-1]
            out = super().forward(x.permute(*new_shape))
            shape = list(np.arange(len(out.shape)))
            new_shape = [0, shape[-1]] + shape[1:-1]
            if self.kernel_size[0] % 2 == 0:
                out = F.pad(out, (0, -1), "constant", 0)
            return out.permute(new_shape)

        return super().forward(x)


def make_cnn(input_size, output_size, hidden_sizes, kernel_size=3):
    """This function used to construct neural network consisting of
       one 1d-convolutional layer that utilizes temporal dependencies,
       fully connected network

    Parameters
    ----------
    input_size : int,
        the dimension of input embeddings

    output_size : int,
        the dimension of out embeddings

    hidden_sizes : tuple,
        the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers,

    kernel_size : int
        kernel size for convolutional layer

    Returns
    -------
    output: tensor
        the processing embeddings
    """
    padding = kernel_size // 2

    cnn_layer = CustomConv1d(input_size, hidden_sizes[0], kernel_size=kernel_size, padding=padding)
    layers = [cnn_layer]

    for i, h in zip(hidden_sizes, hidden_sizes[1:]):
        layers.extend([nn.Linear(i, h), nn.ReLU()])
    if isinstance(output_size, tuple):
        net = nn.Sequential(*layers)
        return [net] + [nn.Linear(hidden_sizes[-1], o) for o in output_size]

    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    return nn.Sequential(*layers)


class GpvaeEncoder(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(128, 128), window_size=24):
        """This module is an encoder with 1d-convolutional network and multivariate Normal posterior used by GP-VAE with
        proposed banded covariance matrix

        Parameters
        ----------
        input_size : int,
            the feature dimension of the input

        z_size : int,
            the feature dimension of the output latent embedding

        hidden_sizes : tuple,
            the tuple of the hidden layer sizes, and the tuple length sets the number of hidden layers

        window_size : int
            the kernel size for the Conv1D layer
        """
        super().__init__()
        self.z_size = int(z_size)
        self.input_size = input_size
        self.net, self.mu_layer, self.logvar_layer = make_cnn(
            input_size, (z_size, z_size * 2), hidden_sizes, window_size
        )

    def forward(self, x):
        mapped = self.net(x)
        batch_size = mapped.size(0)
        time_length = mapped.size(1)

        num_dim = len(mapped.shape)
        mu = self.mu_layer(mapped)
        logvar = self.logvar_layer(mapped)
        mapped_mean = torch.transpose(mu, num_dim - 1, num_dim - 2)
        mapped_covar = torch.transpose(logvar, num_dim - 1, num_dim - 2)
        mapped_covar = torch.sigmoid(mapped_covar)
        mapped_reshaped = mapped_covar.reshape(batch_size, self.z_size, 2 * time_length)

        dense_shape = [batch_size, self.z_size, time_length, time_length]
        idxs_1 = np.repeat(np.arange(batch_size), self.z_size * (2 * time_length - 1))
        idxs_2 = np.tile(np.repeat(np.arange(self.z_size), (2 * time_length - 1)), batch_size)
        idxs_3 = np.tile(
            np.concatenate([np.arange(time_length), np.arange(time_length - 1)]),
            batch_size * self.z_size,
        )
        idxs_4 = np.tile(
            np.concatenate([np.arange(time_length), np.arange(1, time_length)]),
            batch_size * self.z_size,
        )
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)

        mapped_values = mapped_reshaped[:, :, :-1].reshape(-1)
        prec_sparse = torch.sparse_coo_tensor(
            torch.LongTensor(idxs_all).t().to(mapped.device),
            (mapped_values).to(mapped.device),
            (dense_shape),
        )
        prec_sparse = prec_sparse.coalesce()
        prec_tril = prec_sparse.to_dense()
        eye = (
            torch.eye(prec_tril.shape[-1])
            .unsqueeze(0)
            .repeat(prec_tril.shape[0], prec_tril.shape[1], 1, 1)
            .to(mapped.device)
        )
        prec_tril = prec_tril + eye
        cov_tril = torch.linalg.solve_triangular(prec_tril, eye, upper=True)
        cov_tril = torch.where(torch.isfinite(cov_tril), cov_tril, torch.zeros_like(cov_tril)).to(mapped.device)

        num_dim = len(cov_tril.shape)
        cov_tril_lower = torch.transpose(cov_tril, num_dim - 1, num_dim - 2)

        z_dist = torch.distributions.MultivariateNormal(loc=mapped_mean, scale_tril=cov_tril_lower)
        return z_dist


class GpvaeDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(256, 256)):
        """This module is a decoder with Gaussian output distribution.

        Parameters
        ----------
        output_size : int,
            the feature dimension of the output

        hidden_sizes: tuple
            the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers.
        """
        super().__init__()
        self.net = make_nn(input_size, output_size, hidden_sizes)

    def forward(self, x):
        mu = self.net(x)
        var = torch.ones_like(mu)
        return torch.distributions.Normal(mu, var)
