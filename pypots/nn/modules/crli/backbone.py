"""Implementation of pypots.nn.modules.crli.backbone."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
from torch import nn

from .layers import CrliDecoder, CrliDiscriminator, CrliGenerator


class BackboneCRLI(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_generator_layers: int,
        rnn_hidden_size: int,
        decoder_fcn_output_dims: list | None,
        rnn_cell_type: str = "GRU",
    ):
        super().__init__()
        self.generator = CrliGenerator(n_generator_layers, n_features, rnn_hidden_size, rnn_cell_type)
        self.discriminator = CrliDiscriminator(rnn_cell_type, n_features)
        self.decoder = CrliDecoder(
            n_steps, rnn_hidden_size * 2, n_features, decoder_fcn_output_dims
        )  # fully connected network is included in Decoder

    def forward(self, X, missing_mask) -> tuple[torch.Tensor, ...]:
        imputation_latent, generator_fb_hidden_states = self.generator(X, missing_mask)
        discrimination = self.discriminator(X, missing_mask, imputation_latent)
        reconstruction, fcn_latent = self.decoder(generator_fb_hidden_states)
        return imputation_latent, discrimination, reconstruction, fcn_latent
