"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Tuple

import torch
import torch.nn as nn

from .layers import (
    GMMLayer,
    PeepholeLSTMCell,
    ImplicitImputation,
)


class BackboneVaDER(nn.Module):
    """

    Parameters
    ----------
    n_steps :
    d_input :
    n_clusters :
    d_rnn_hidden :
    d_mu_stddev :
    eps :
    alpha :
        Weight of the latent loss.
        The final loss = `alpha`*latent loss + reconstruction loss


    Attributes
    ----------

    """

    def __init__(
        self,
        n_steps: int,
        d_input: int,
        n_clusters: int,
        d_rnn_hidden: int,
        d_mu_stddev: int,
        eps: float = 1e-9,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.d_input = d_input
        self.n_clusters = n_clusters
        self.d_rnn_hidden = d_rnn_hidden
        self.d_mu_stddev = d_mu_stddev
        self.eps = eps
        self.alpha = alpha

        # building model components
        self.implicit_imputation_layer = ImplicitImputation(d_input)
        self.encoder = PeepholeLSTMCell(d_input, d_rnn_hidden)
        self.decoder = PeepholeLSTMCell(d_input, d_rnn_hidden)
        self.ae_encode_layers = nn.Sequential(nn.Linear(d_rnn_hidden, d_rnn_hidden), nn.Softplus())
        self.ae_decode_layers = nn.Sequential(nn.Linear(d_mu_stddev, d_rnn_hidden), nn.Softplus())
        self.mu_layer = nn.Linear(d_rnn_hidden, d_mu_stddev)  # layer for mean
        self.stddev_layer = nn.Linear(d_rnn_hidden, d_mu_stddev)  # layer for standard variance
        self.rnn_transform_layer = nn.Linear(d_rnn_hidden, d_input)
        self.gmm_layer = GMMLayer(d_mu_stddev, n_clusters)

    @staticmethod
    def z_sampling(
        mu_tilde: torch.Tensor,
        stddev_tilde: torch.Tensor,
    ) -> torch.Tensor:
        noise = mu_tilde.data.new(mu_tilde.size()).normal_()
        z = torch.add(mu_tilde, torch.exp(0.5 * stddev_tilde) * noise)
        return z

    def encode(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = X.size(0)

        X_imputed = self.implicit_imputation_layer(X, missing_mask)

        hidden_state = torch.zeros((batch_size, self.d_rnn_hidden), dtype=X.dtype, device=X.device)
        cell_state = torch.zeros((batch_size, self.d_rnn_hidden), dtype=X.dtype, device=X.device)
        # cell_state_collector = torch.empty((batch_size, self.n_steps, self.d_rnn_hidden),
        #                                    dtype=X.dtype, device=X.device)
        for i in range(self.n_steps):
            x = X_imputed[:, i, :]
            hidden_state, cell_state = self.encoder(x, (hidden_state, cell_state))
            # cell_state_collector[:, i, :] = cell_state

        cell_state_collector = self.ae_encode_layers(cell_state)
        mu_tilde = self.mu_layer(cell_state_collector)
        stddev_tilde = self.stddev_layer(cell_state_collector)
        z = self.z_sampling(mu_tilde, stddev_tilde)
        return z, mu_tilde, stddev_tilde

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden_state = z
        hidden_state = self.ae_decode_layers(hidden_state)

        cell_state = torch.zeros(hidden_state.size(), dtype=z.dtype, device=z.device)
        inputs = torch.zeros((z.size(0), self.n_steps, self.d_input), dtype=z.dtype, device=z.device)

        hidden_state_collector = torch.empty(
            (z.size(0), self.n_steps, self.d_rnn_hidden), dtype=z.dtype, device=z.device
        )
        for i in range(self.n_steps):
            x = inputs[:, i, :]
            hidden_state, cell_state = self.decoder(x, (hidden_state, cell_state))
            hidden_state_collector[:, i, :] = hidden_state

        reconstruction = self.rnn_transform_layer(hidden_state_collector)
        return reconstruction

    def forward(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu_tilde, stddev_tilde = self.encode(X, missing_mask)
        X_reconstructed = self.decode(z)
        mu_c, var_c, phi_c = self.gmm_layer()
        return X_reconstructed, mu_c, var_c, phi_c, z, mu_tilde, stddev_tilde
