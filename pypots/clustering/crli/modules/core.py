"""
The implementation of CRLI (Clustering Representation Learning on Incomplete time-series data) for
the partially-observed time-series clustering task.

Refer to the paper "Ma, Q., Chen, C., Li, S., & Cottrell, G. W. (2021).
Learning Representations for Incomplete Time Series Clustering. AAAI 2021."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .submodules import Generator, Decoder, Discriminator
from ....utils.metrics import cal_mse


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
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.generator = Generator(
            n_generator_layers, n_features, rnn_hidden_size, rnn_cell_type, device
        )
        self.discriminator = Discriminator(rnn_cell_type, n_features, device)
        self.decoder = Decoder(
            n_steps, rnn_hidden_size * 2, n_features, decoder_fcn_output_dims, device
        )  # fully connected network is included in Decoder
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,  # FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the
            # value of `n_init` explicitly to suppress the warning.
        )  # TODO: implement KMean with torch for gpu acceleration

        self.n_clusters = n_clusters
        self.lambda_kmeans = lambda_kmeans
        self.device = device

    def cluster(self, inputs: dict, training_object: str = "generator") -> dict:
        # concat final states from generator and input it as the initial state of decoder
        imputation, imputed_X, generator_fb_hidden_states = self.generator(inputs)
        inputs["imputation"] = imputation
        inputs["imputed_X"] = imputed_X
        inputs["generator_fb_hidden_states"] = generator_fb_hidden_states
        if training_object == "discriminator":
            discrimination = self.discriminator(inputs)
            inputs["discrimination"] = discrimination
            return inputs  # if only train discriminator, then no need to run decoder

        reconstruction, fcn_latent = self.decoder(inputs)
        inputs["reconstruction"] = reconstruction
        inputs["fcn_latent"] = fcn_latent
        return inputs

    def forward(
        self,
        inputs: dict,
        training_object: str = "generator",
        training: bool = True,
    ) -> dict:
        assert training_object in [
            "generator",
            "discriminator",
        ], 'training_object should be "generator" or "discriminator"'

        X = inputs["X"]
        missing_mask = inputs["missing_mask"]
        batch_size, n_steps, n_features = X.shape
        losses = {}
        inputs = self.cluster(inputs, training_object)
        if not training:
            # if only run clustering, then no need to calculate loss
            return inputs

        if training_object == "discriminator":
            l_D = F.binary_cross_entropy_with_logits(
                inputs["discrimination"], missing_mask
            )
            losses["discrimination_loss"] = l_D
        else:
            inputs["discrimination"] = inputs["discrimination"].detach()
            l_G = F.binary_cross_entropy_with_logits(
                inputs["discrimination"], 1 - missing_mask, weight=1 - missing_mask
            )
            l_pre = cal_mse(inputs["imputation"], X, missing_mask)
            l_rec = cal_mse(inputs["reconstruction"], X, missing_mask)
            HTH = torch.matmul(inputs["fcn_latent"], inputs["fcn_latent"].permute(1, 0))
            term_F = torch.nn.init.orthogonal_(
                torch.randn(batch_size, self.n_clusters, device=self.device), gain=1
            )
            FTHTHF = torch.matmul(torch.matmul(term_F.permute(1, 0), HTH), term_F)
            l_kmeans = torch.trace(HTH) - torch.trace(FTHTHF)  # k-means loss
            loss_gene = l_G + l_pre + l_rec + l_kmeans * self.lambda_kmeans
            losses["generation_loss"] = loss_gene
        return losses
