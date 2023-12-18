"""
The implementation of SAITS for the partially-observed time-series imputation task.

Refer to the paper "Du, W., Cote, D., & Liu, Y. (2023). SAITS: Self-Attention-based Imputation for Time Series.
Expert systems with applications."

Notes
-----
Partial implementation uses code from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....nn.modules.transformer import EncoderLayer, PositionalEncoding
from ....utils.metrics import calc_mae


class _SAITS(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_steps: int,
        n_features: int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
        diagonal_attention_mask: bool = True,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        customized_loss_func: Callable = calc_mae,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        # concatenate the feature vector and missing mask, hence double the number of features
        actual_n_features = n_features * 2
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.customized_loss_func = customized_loss_func

        self.layer_stack_for_first_block = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_stack_for_second_block = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
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

    def _process(
        self,
        inputs: dict,
        diagonal_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list, list]:
        X, masks = inputs["X"], inputs["missing_mask"]

        # first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2)
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely, term e in the math equation
        first_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, first_DMSA_attn_weights = encoder_layer(
                enc_output, diagonal_attention_mask
            )

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        input_X_for_second = torch.cat([X_prime, masks], dim=2)
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        second_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, second_DMSA_attn_weights = encoder_layer(
                enc_output, diagonal_attention_mask
            )

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
            self.weight_combine(torch.cat([masks, copy_second_DMSA_weights], dim=2))
        )
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # replace non-missing part with original data
        X_c = masks * X + (1 - masks) * X_tilde_3

        return (
            X_c,
            [X_tilde_1, X_tilde_2, X_tilde_3],
            [first_DMSA_attn_weights, second_DMSA_attn_weights, combining_weights],
        )

    def forward(
        self,
        inputs: dict,
        diagonal_attention_mask: bool = False,
        training: bool = True,
    ) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        if (training and self.diagonal_attention_mask) or (
            (not training) and diagonal_attention_mask
        ):
            diagonal_attention_mask = (1 - torch.eye(self.n_steps)).to(X.device)
            # then broadcast on the batch axis
            diagonal_attention_mask = diagonal_attention_mask.unsqueeze(0)
        else:
            diagonal_attention_mask = None

        (
            imputed_data,
            [X_tilde_1, X_tilde_2, X_tilde_3],
            [first_DMSA_attn_weights, second_DMSA_attn_weights, combining_weights],
        ) = self._process(inputs, diagonal_attention_mask)

        results = {
            "first_DMSA_attn_weights": first_DMSA_attn_weights,
            "second_DMSA_attn_weights": second_DMSA_attn_weights,
            "combining_weights": combining_weights,
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            ORT_loss = 0
            ORT_loss += self.customized_loss_func(X_tilde_1, X, masks)
            ORT_loss += self.customized_loss_func(X_tilde_2, X, masks)
            ORT_loss += self.customized_loss_func(X_tilde_3, X, masks)
            ORT_loss /= 3

            MIT_loss = self.customized_loss_func(
                X_tilde_3, inputs["X_ori"], inputs["indicating_mask"]
            )

            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss
            # `loss` is always the item for backward propagating to update the model
            loss = self.ORT_weight * ORT_loss + self.MIT_weight * MIT_loss
            results["loss"] = loss

        return results
