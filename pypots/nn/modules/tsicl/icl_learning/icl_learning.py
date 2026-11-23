"""
Vendored from the TS-ICL model architecture: https://github.com/EDF-Lab/ts-icl
"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause
#
# Portions of this file are adapted from the following project(s), redistributed under
# their original license terms reproduced below:
#   - TabICL: https://github.com/soda-inria/tabicl
#
# ---- TabICL (https://github.com/soda-inria/tabicl) ----
#
# BSD 3-Clause License
#
# Copyright (c) 2025, Soda team @ Inria
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from ..encoder.blocks.attention import CrossAttention, FeedForward
from ..encoder.blocks.utils import PreNorm, PreNormCross
from .encoders import Encoder

# ACKNOWLEDGEMENT: code adapted from TabICL implementation:
# https://github.com/soda-inria/tabicl/tree/main


class ICLearning(nn.Module):
    """In-context learning head of TS-ICL, regressing the quantiles of a series at query time coordinates.

    This module implements in-context learning that:
    1. Takes the (time-localized) latent representations of all context and query coordinates, and the
       observed values at the context coordinates as input
    2. Conditions the model on the context (observed) points
    3. Predicts the quantiles of the series at the query coordinates based on learned patterns

    Parameters
    ----------
    d_model : int
        Model dimension

    num_blocks : int
        Number of blocks used in the ICL encoder (MHA + RoPE)

    nhead : int
        Number of attention heads of the ICL encoder

    dim_feedforward : int
        Dimension of the feedforward network of the ICL encoder

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
        start_quantile: float = 0.05,
        end_quantile: float = 0.95,
        nb_quantiles: int = 19,
        quantile_median: int = 9,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        if self.norm_first:
            self.ln = nn.LayerNorm(d_model)

        self.y_encoder = nn.Linear(1, d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, nb_quantiles))

        self.start_quantile = start_quantile
        self.end_quantile = end_quantile
        self.nb_quantiles = nb_quantiles
        self.quantile_median = quantile_median
        self.d_model = d_model

    def _icl_predictions(self, R: Tensor, y_train: Tensor) -> Tensor:
        """In-context learning predictions.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data
        """

        train_size = y_train.shape[1]
        R[:, :train_size] = R[:, :train_size] + self.y_encoder(y_train.float())
        src = self.tf_icl(R, attn_mask=train_size)
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)  # (B, T, 1)

        return out

    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        y_cov: Tensor | None = None,
    ) -> Tensor:
        """In-context learning based on learned row representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data

        Returns
        -------
        Tensor
              Predictions of shape (B, T-train_size, 1), which will be further handled by the training code.
        """

        train_size = y_train.shape[1]
        out = self._icl_predictions(R, y_train)
        out = out[:, train_size:]

        return out


class ICLearningCrossAttn(nn.Module):
    """In-context learning head of TS-ICL with covariate cross-attention, regressing the quantiles of a series
    at query time coordinates.

    This module implements in-context learning that:
    1. Takes the (time-localized) latent representations of all context and query coordinates, and the
       observed values at the context coordinates as input
    2. Conditions the model on the context (observed) points
    3. Predicts the quantiles of the series at the query coordinates based on learned patterns

    Parameters
    ----------
    d_model : int
        Model dimension

    num_blocks : int
        Number of blocks used in the ICL encoder (MHA + RoPE)

    nhead : int
        Number of attention heads of the ICL encoder

    dim_feedforward : int
        Dimension of the feedforward network of the ICL encoder

    cross_heads : int
        Number of cross-attn heads used to embed the values and covar

    cross_dim_head : int
        Dim of each cross-attn head used to embed the values and covar

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        cross_heads: int = 4,
        cross_dim_head: int = 64,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
        start_quantile: float = 0.05,
        end_quantile: float = 0.95,
        nb_quantiles: int = 19,
        quantile_median: int = 9,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        if self.norm_first:
            self.ln = nn.LayerNorm(d_model)

        # value projector:
        self.y_encoder = nn.Linear(1, d_model)

        # covar projector:
        self.cov_encoder = nn.Linear(1, d_model)

        # learnable tokens:
        self.target_token = nn.Parameter(torch.randn(1, d_model) * 0.05)
        self.sep_token = nn.Parameter(torch.randn(2, d_model) * 0.05)

        # cross-attn layer:
        self.cross_attn = nn.ModuleList(
            [
                PreNormCross(
                    d_model,
                    CrossAttention(
                        query_dim=d_model,
                        key_dim=d_model,
                        value_dim=d_model,
                        heads=cross_heads,
                        dim_head=cross_dim_head,
                        headwise_attn_output_gate=False,
                    ),
                    k_dim=d_model,
                    v_dim=d_model,
                ),
                PreNorm(d_model, FeedForward(d_model)),
            ]
        )
        # quantile head:
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, nb_quantiles))

        self.start_quantile = start_quantile
        self.end_quantile = end_quantile
        self.nb_quantiles = nb_quantiles
        self.quantile_median = quantile_median
        self.d_model = d_model

    def _icl_predictions(
        self,
        R: Tensor,
        y_train: Tensor,
        y_cov: Tensor | None = None,
    ) -> Tensor:
        """In-context learning predictions.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data
        """

        train_size = y_train.shape[1]

        bs = len(y_train)

        # prepare queries:
        x = repeat(self.target_token, "1 d -> (b T) 1 d", b=bs, T=R.shape[1])  # (bs x T, 1, d)

        # prepare keys, values (on context):
        y = self.y_encoder(y_train.float())  # (bs, T_ctx, d)
        train_seq = [
            rearrange(y, "b T d -> (b T) 1 d"),  # (bs x T_ctx, 1, d)
            repeat(self.sep_token, "m d -> (b T) m d", b=bs, T=train_size),  # (bs x T_ctx, 2, d)
        ]

        # prepare keys, values (on target):
        test_seq = [repeat(self.sep_token, "m d -> (b T) m d", b=bs, T=R.shape[1] - train_size)]  # (bs x T_mis, 2, d)

        attn_mask_train, attn_mask_test = None, None

        # append covar to keys, values:
        if y_cov is not None:
            assert y_cov.shape[-2] >= train_size

            # get mask of observed covariates:
            is_observed_mask = ~torch.isnan(y_cov)  # (bs, c, T, 1)

            # replace nan (missing covars) by zeroes:
            # (the replacing value does not matter, will be zeroed-out by attn mask)
            y_cov = torch.nan_to_num(y_cov, nan=0.0)

            # project covariates to latent space:
            z = self.cov_encoder(y_cov)  # (bs, C, T, d)

            # complete TF-ICL train sequence:
            train_seq.append(rearrange(z[:, :, :train_size, :], "b c t d -> (b t) c d", b=bs, t=train_size))

            # if missing covar in train sequence, build attention mask:
            if (~is_observed_mask[..., :train_size, :]).sum() > 0:
                attn_mask_train = torch.cat(
                    [
                        torch.ones(len(train_seq[0]), 1 + train_seq[1].shape[1], 1)
                        .bool()
                        .to(z.device),  # (bs x T_ctx, 1+2, 1)
                        rearrange(is_observed_mask[..., :train_size, :], "b c T 1 -> (b T) c 1"),  # (bs x T_ctx, C, 1)
                    ],
                    dim=1,
                )  # (bs x T_ctx, 1+2+C, 1)

            if z.shape[-2] > train_size:
                # complete TF-ICL test sequence:
                test_seq.append(
                    rearrange(z[:, :, train_size:, :], "b c t d -> (b t) c d", b=bs, t=R.shape[1] - train_size)
                )

                # if missing covar in test sequence, build attention mask:
                if (~is_observed_mask[..., train_size:, :]).sum() > 0:
                    attn_mask_test = torch.cat(
                        [
                            torch.ones(len(test_seq[0]), test_seq[0].shape[1], 1)
                            .bool()
                            .to(z.device),  # (bs x T_mis, 2, 1)
                            rearrange(
                                is_observed_mask[..., train_size:, :], "b c T 1 -> (b T) c 1"
                            ),  # (bs x T_mis, C, 1)
                        ],
                        dim=1,
                    )  # (bs x T_mis, 2+C, 1)

        # build sequence of KV:
        train_seq = torch.cat(train_seq, dim=1)  # (bs x T_ctx, 1+2+C, d)
        test_seq = torch.cat(test_seq, dim=1)  # type: ignore # (bs x T_mis, 2+C, d)

        # do cross-attn on train and test separately:
        cross_attn, cross_ff = self.cross_attn
        # query_tr: x [(b T_tr) 1 d]
        # kv_tr: [(b T_tr) 1+2+C d]
        # mask: (b T_tr) 1+2+C
        x_tr = (
            cross_attn(x[: len(train_seq)], k=train_seq, v=train_seq, mask=attn_mask_train) + x[: len(train_seq)]
        )  # (bs x T_ctx, 1, d)
        x_tr = cross_ff(x_tr) + x_tr
        x_tr = rearrange(x_tr, "(b T) 1 d -> b T d", b=bs)

        x_te = (
            cross_attn(x[: len(test_seq)], k=test_seq, v=test_seq, mask=attn_mask_test) + x[: len(test_seq)]
        )  # (bs x T_mis, 1, d)
        x_te = cross_ff(x_te) + x_te
        x_te = rearrange(x_te, "(b T) 1 d -> b T d", b=bs)

        # update feature tensor:
        R[:, :train_size] = R[:, :train_size] + x_tr
        R[:, train_size:] = R[:, train_size:] + x_te

        # to tf icl:
        src = self.tf_icl(R, attn_mask=train_size)
        if self.norm_first:
            src = self.ln(src)

        # quantile head:
        out = self.decoder(src)  # (B, T, 1)

        return out

    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        y_cov: Tensor | None = None,
    ) -> Tensor:
        """In-context learning based on learned row representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data

        Returns
        -------
        Tensor
              Predictions of shape (B, T-train_size, 1), which will be further handled by the training code.
        """

        train_size = y_train.shape[1]
        if y_cov is not None:
            assert (y_cov.shape[2] == train_size) or (y_cov.shape[2] == R.shape[1])
        out = self._icl_predictions(R, y_train, y_cov)
        out = out[:, train_size:]

        return out
