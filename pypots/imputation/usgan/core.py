"""
The core wrapper assembles the submodules of USGAN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.usgan import BackboneUSGAN


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
    ):
        super().__init__()
        self.backbone = BackboneUSGAN(
            n_steps,
            n_features,
            rnn_hidden_size,
            lambda_mse,
            hint_rate,
            dropout_rate,
        )

    def forward(
        self,
        inputs: dict,
        training_object: str = "generator",
    ) -> dict:
        assert training_object in [
            "generator",
            "discriminator",
        ], 'training_object should be "generator" or "discriminator"'

        results = {}
        if self.training:
            if training_object == "discriminator":
                imputed_data, reconstruction, discrimination_loss = self.backbone(inputs, training_object)
                loss = discrimination_loss
            else:
                imputed_data, reconstruction, generation_loss = self.backbone(inputs, training_object)
                loss = generation_loss
            results["loss"] = loss
        else:
            imputed_data, reconstruction = self.backbone(inputs, training_object)

        results["imputation"] = imputed_data
        results["reconstruction"] = reconstruction
        return results
