"""
The core wrapper assembles the submodules of VaDER clustering model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import numpy as np
import torch
import torch.nn as nn

from ...nn.functional import calc_mse
from ...nn.modules.vader import BackboneVaDER


def inverse_softplus(x: np.ndarray) -> np.ndarray:
    b = x < 1e2
    x[b] = np.log(np.exp(x[b]) - 1.0 + 1e-9)
    return x


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

        self.backbone = BackboneVaDER(
            n_steps,
            d_input,
            n_clusters,
            d_rnn_hidden,
            d_mu_stddev,
            eps,
            alpha,
        )

    def forward(
        self,
        inputs: dict,
        pretrain: bool = False,
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
        ) = self.backbone(X, missing_mask)

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
        reconstruction_loss = unscaled_reconstruction_loss * self.n_steps * self.d_input / missing_mask.sum()

        if pretrain:
            results["loss"] = reconstruction_loss
            return results

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
        log_pdf_z = log_pdf_z.reshape([batch_size, self.n_clusters, self.d_mu_stddev])

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

        latent_loss1 = 0.5 * torch.sum(gamma_c * torch.sum(term1 + term2 + term3, dim=2), dim=1)
        latent_loss2 = -torch.sum(gamma_c * (log_phi_c - log_gamma_c), dim=1)
        latent_loss3 = -0.5 * torch.sum(1 + stddev_tilde, dim=1)

        latent_loss1 = latent_loss1.mean()
        latent_loss2 = latent_loss2.mean()
        latent_loss3 = latent_loss3.mean()
        latent_loss = latent_loss1 + latent_loss2 + latent_loss3

        results["loss"] = reconstruction_loss + self.alpha * latent_loss

        return results
