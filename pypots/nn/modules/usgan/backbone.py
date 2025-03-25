"""

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import UsganDiscriminator
from ..brits import BackboneBRITS
from ....nn.functional import calc_mse


class BackboneUSGAN(nn.Module):
    """USGAN model"""

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        lambda_mse: float,
        hint_rate: float = 0.7,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse

        self.generator = BackboneBRITS(n_steps, n_features, rnn_hidden_size)
        self.discriminator = UsganDiscriminator(
            n_features,
            rnn_hidden_size,
            hint_rate,
            dropout_rate,
        )

    def forward(
        self,
        inputs: dict,
        training_object: str = "generator",
    ) -> Tuple[torch.Tensor, ...]:
        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            _,
            _,
            _,
            _,
        ) = self.generator(inputs)

        reconstruction = (f_reconstruction + b_reconstruction) / 2

        # if in training mode, return results with losses
        if self.training:
            forward_X = inputs["forward"]["X"]
            forward_missing_mask = inputs["forward"]["missing_mask"]

            if training_object == "discriminator":
                discrimination = self.discriminator(imputed_data.detach(), forward_missing_mask)
                l_D = F.binary_cross_entropy_with_logits(discrimination, forward_missing_mask)
                discrimination_loss = l_D
                return imputed_data, reconstruction, discrimination_loss
            else:
                discrimination = self.discriminator(imputed_data, forward_missing_mask)
                l_G = -F.binary_cross_entropy_with_logits(
                    discrimination,
                    forward_missing_mask,
                    weight=1 - forward_missing_mask,
                )
                reconstruction = (f_reconstruction + b_reconstruction) / 2
                reconstruction_loss = calc_mse(forward_X, reconstruction, forward_missing_mask) + 0.1 * calc_mse(
                    f_reconstruction, b_reconstruction
                )
                loss_gene = l_G + self.lambda_mse * reconstruction_loss
                generation_loss = loss_gene
                return imputed_data, reconstruction, generation_loss
        else:
            return imputed_data, reconstruction
