"""
The implementation of VaDER for the partially-observed time-series clustering task.

Refer to the paper "Jong, J.D., Emon, M.A., Wu, P., Karki, R., Sood, M., Godard, P., Ahmad, A., Vrooman, H.A.,
Hofmann-Apitius, M., & Fröhlich, H. (2019).
Deep learning for clustering of multivariate clinical patient trajectories with missing values. GigaScience."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Tuple

import torch
import torch.nn as nn

from .submodules import (
    GMMLayer,
    PeepholeLSTMCell,
    ImplicitImputation,
)
from ....utils.metrics import calc_mse


class _VaDER(nn.Module):
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
        self.ae_encode_layers = nn.Sequential(
            nn.Linear(d_rnn_hidden, d_rnn_hidden), nn.Softplus()
        )
        self.ae_decode_layers = nn.Sequential(
            nn.Linear(d_mu_stddev, d_rnn_hidden), nn.Softplus()
        )
        self.mu_layer = nn.Linear(d_rnn_hidden, d_mu_stddev)  # layer for mean
        self.stddev_layer = nn.Linear(
            d_rnn_hidden, d_mu_stddev
        )  # layer for standard variance
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

        hidden_state = torch.zeros(
            (batch_size, self.d_rnn_hidden), dtype=X.dtype, device=X.device
        )
        cell_state = torch.zeros(
            (batch_size, self.d_rnn_hidden), dtype=X.dtype, device=X.device
        )
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
        inputs = torch.zeros(
            (z.size(0), self.n_steps, self.d_input), dtype=z.dtype, device=z.device
        )

        hidden_state_collector = torch.empty(
            (z.size(0), self.n_steps, self.d_rnn_hidden), dtype=z.dtype, device=z.device
        )
        for i in range(self.n_steps):
            x = inputs[:, i, :]
            hidden_state, cell_state = self.decoder(x, (hidden_state, cell_state))
            hidden_state_collector[:, i, :] = hidden_state

        reconstruction = self.rnn_transform_layer(hidden_state_collector)
        return reconstruction

    def get_results(
        self, X: torch.Tensor, missing_mask: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        z, mu_tilde, stddev_tilde = self.encode(X, missing_mask)
        X_reconstructed = self.decode(z)
        mu_c, var_c, phi_c = self.gmm_layer()
        return X_reconstructed, mu_c, var_c, phi_c, z, mu_tilde, stddev_tilde

    def forward(
        self,
        inputs: dict,
        pretrain: bool = False,
        training: bool = True,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        device = X.device

        (
            X_reconstructed,
            mu_c,
            var_c,
            phi_c,
            z,
            mu_tilde,
            stddev_tilde,
        ) = self.get_results(X, missing_mask)

        results = {
            "mu_tilde": mu_tilde,
            "stddev_tilde": stddev_tilde,
            "mu": mu_c,
            "var": var_c,
            "phi": phi_c,
            "z": z,
            "imputation_latent": X_reconstructed,
        }

        # calculate the reconstruction loss
        unscaled_reconstruction_loss = calc_mse(X_reconstructed, X, missing_mask)
        reconstruction_loss = (
            unscaled_reconstruction_loss
            * self.n_steps
            * self.d_input
            / missing_mask.sum()
        )

        if pretrain:
            results["loss"] = reconstruction_loss
            return results

        # if in training mode, return results with losses
        if training:
            # calculate the latent loss for model training
            var_tilde = torch.exp(stddev_tilde)
            stddev_c = torch.log(var_c + self.eps)
            log_2pi = torch.log(torch.tensor([2 * torch.pi], device=device))
            log_phi_c = torch.log(phi_c + self.eps)

            batch_size = z.shape[0]

            ii, jj = torch.meshgrid(
                torch.arange(self.n_clusters, dtype=torch.int64, device=device),
                torch.arange(batch_size, dtype=torch.int64, device=device),
                indexing="ij",
            )
            ii = ii.flatten()
            jj = jj.flatten()

            lsc_b = stddev_c.index_select(dim=0, index=ii)
            mc_b = mu_c.index_select(dim=0, index=ii)
            sc_b = var_c.index_select(dim=0, index=ii)
            z_b = z.index_select(dim=0, index=jj)
            log_pdf_z = -0.5 * (lsc_b + log_2pi + torch.square(z_b - mc_b) / sc_b)
            log_pdf_z = log_pdf_z.reshape(
                [batch_size, self.n_clusters, self.d_mu_stddev]
            )

            log_p = log_phi_c + log_pdf_z.sum(dim=2)
            lse_p = log_p.logsumexp(dim=1, keepdim=True)
            log_gamma_c = log_p - lse_p
            gamma_c = torch.exp(log_gamma_c)

            term1 = torch.log(var_c + self.eps)
            st_b = var_tilde.index_select(dim=0, index=jj)
            sc_b = var_c.index_select(dim=0, index=ii)
            term2 = torch.reshape(
                st_b / (sc_b + self.eps),
                [batch_size, self.n_clusters, self.d_mu_stddev],
            )
            mt_b = mu_tilde.index_select(dim=0, index=jj)
            mc_b = mu_c.index_select(dim=0, index=ii)
            term3 = torch.reshape(
                torch.square(mt_b - mc_b) / (sc_b + self.eps),
                [batch_size, self.n_clusters, self.d_mu_stddev],
            )

            latent_loss1 = 0.5 * torch.sum(
                gamma_c * torch.sum(term1 + term2 + term3, dim=2), dim=1
            )
            latent_loss2 = -torch.sum(gamma_c * (log_phi_c - log_gamma_c), dim=1)
            latent_loss3 = -0.5 * torch.sum(1 + stddev_tilde, dim=1)

            latent_loss1 = latent_loss1.mean()
            latent_loss2 = latent_loss2.mean()
            latent_loss3 = latent_loss3.mean()
            latent_loss = latent_loss1 + latent_loss2 + latent_loss3

            results["loss"] = reconstruction_loss + self.alpha * latent_loss

        return results
