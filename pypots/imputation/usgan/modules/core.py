"""
The implementation of USGAN for the partially-observed time-series imputation task.

Refer to the paper "Miao, X., Wu, Y., Wang, J., Gao, Y., Mao, X., & Yin, J. (2021).
Generative Semi-supervised Learning for Multivariate Time Series Imputation. AAAI 2021."

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodules import Discriminator
from ...brits.modules import _BRITS


class _USGAN(nn.Module):
    """USGAN model"""

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        lambda_mse: float,
        hint_rate: float = 0.7,
        dropout_rate: float = 0.0,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.generator = _BRITS(n_steps, n_features, rnn_hidden_size, device)
        self.discriminator = Discriminator(
            n_features,
            rnn_hidden_size,
            hint_rate=hint_rate,
            dropout_rate=dropout_rate,
            device=device,
        )

        self.lambda_mse = lambda_mse
        self.device = device

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

        results = self.generator(inputs, training=training)

        # if in training mode, return results with losses
        if training:
            forward_X = inputs["forward"]["X"]
            forward_missing_mask = inputs["forward"]["missing_mask"]

            inputs["discrimination"] = self.discriminator(
                forward_X, forward_missing_mask
            )

            if training_object == "discriminator":
                l_D = F.binary_cross_entropy_with_logits(
                    inputs["discrimination"], forward_missing_mask
                )
                results["discrimination_loss"] = l_D
            else:
                inputs["discrimination"] = inputs["discrimination"].detach()
                l_G = F.binary_cross_entropy_with_logits(
                    inputs["discrimination"],
                    1 - forward_missing_mask,
                    weight=1 - forward_missing_mask,
                )
                loss_gene = l_G + self.lambda_mse * results["loss"]
                results["generation_loss"] = loss_gene

        return results
