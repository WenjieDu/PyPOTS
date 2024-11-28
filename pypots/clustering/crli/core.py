"""
The core wrapper assembles the submodules of CRLI clustering model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from ...nn.functional import calc_mse
from ...nn.modules.crli import BackboneCRLI


class _CRLI(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_clusters: int,
        n_generator_layers: int,
        rnn_hidden_size: int,
        decoder_fcn_output_dims: Optional[list],
        lambda_kmeans: float,
        rnn_cell_type: str = "GRU",
    ):
        super().__init__()

        self.backbone = BackboneCRLI(
            n_steps,
            n_features,
            n_generator_layers,
            rnn_hidden_size,
            decoder_fcn_output_dims,
            rnn_cell_type,
        )

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,  # FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the
            # value of `n_init` explicitly to suppress the warning.
        )
        self.term_F = None
        self.counter_for_updating_F = 0

        self.n_clusters = n_clusters
        self.lambda_kmeans = lambda_kmeans

    def forward(
        self,
        inputs: dict,
        training_object: str = "generator",
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        imputation_latent, discrimination, reconstruction, fcn_latent = self.backbone(X, missing_mask)
        results = {
            "imputation_latent": imputation_latent,
            "discrimination": discrimination,
            "reconstruction": reconstruction,
            "fcn_latent": fcn_latent,
        }

        if training_object == "discriminator":
            l_D = F.binary_cross_entropy_with_logits(discrimination, missing_mask)
            results["discrimination_loss"] = l_D
        else:
            # discrimination = discrimination.detach()
            l_G = F.binary_cross_entropy_with_logits(discrimination, 1 - missing_mask, weight=1 - missing_mask)
            l_pre = calc_mse(imputation_latent, X, missing_mask)
            l_rec = calc_mse(reconstruction, X, missing_mask)
            HTH = torch.matmul(fcn_latent, fcn_latent.permute(1, 0))

            if self.counter_for_updating_F == 0 or self.counter_for_updating_F % 10 == 0:
                U, s, V = torch.linalg.svd(fcn_latent)
                self.term_F = U[:, : self.n_clusters]

            FTHTHF = torch.matmul(torch.matmul(self.term_F.permute(1, 0), HTH), self.term_F)
            l_kmeans = torch.trace(HTH) - torch.trace(FTHTHF)  # k-means loss
            loss_gene = l_G + l_pre + l_rec + l_kmeans * self.lambda_kmeans
            results["generation_loss"] = loss_gene

        return results
