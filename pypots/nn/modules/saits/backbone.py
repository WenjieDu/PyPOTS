"""
Refer to the paper "Du, W., Cote, D., & Liu, Y. (2023). SAITS: Self-Attention-based Imputation for Time Series.
Expert systems with applications."

Notes
-----
Partial implementation uses code from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformer import (
    PositionalEncoding,
    TransformerEncoderLayer,
    ScaledDotProductAttention,
)


class BackboneSAITS(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_steps: int,
        n_features: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()

        # concatenate the feature vector and missing mask, hence double the number of features
        actual_n_features = n_features * 2

        self.layer_stack_for_first_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    d_ffn,
                    n_heads,
                    d_k,
                    d_v,
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_stack_for_second_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    d_ffn,
                    n_heads,
                    d_k,
                    d_v,
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_positions=n_steps)
        # for the 1st block
        self.embedding_1 = nn.Linear(actual_n_features, d_model)
        self.reduce_dim_z = nn.Linear(d_model, n_features)
        # for the 2nd block
        self.embedding_2 = nn.Linear(actual_n_features, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, n_features)
        self.reduce_dim_gamma = nn.Linear(n_features, n_features)
        # for delta decay factor
        self.weight_combine = nn.Linear(n_features + n_steps, n_features)

    def forward(
        self, X, missing_mask, attn_mask: Optional = None
    ) -> Tuple[torch.Tensor, ...]:

        # first DMSA block
        input_X_for_first = torch.cat([X, missing_mask], dim=2)
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely, term e in the math equation
        first_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, first_DMSA_attn_weights = encoder_layer(enc_output, attn_mask)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = missing_mask * X + (1 - missing_mask) * X_tilde_1

        # second DMSA block
        input_X_for_second = torch.cat([X_prime, missing_mask], dim=2)
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        second_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, second_DMSA_attn_weights = encoder_layer(enc_output, attn_mask)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        copy_second_DMSA_weights = second_DMSA_attn_weights.clone()
        copy_second_DMSA_weights = copy_second_DMSA_weights.squeeze(
            dim=1
        )  # namely term A_hat in Eq.
        if len(copy_second_DMSA_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 3)
            copy_second_DMSA_weights = copy_second_DMSA_weights.mean(dim=3)
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 2)

        # namely term eta
        combining_weights = torch.sigmoid(
            self.weight_combine(
                torch.cat([missing_mask, copy_second_DMSA_weights], dim=2)
            )
        )
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1

        return (
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_DMSA_attn_weights,
            second_DMSA_attn_weights,
            combining_weights,
        )
