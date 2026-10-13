"""
Vendored from the TS-ICL model architecture: https://github.com/EDF-Lab/ts-icl
"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause
#
# Portions of this file are adapted from the following project(s), redistributed under
# their original license terms reproduced below:
#   - AROMA: https://github.com/LouisSerrano/aroma
#   - perceiver-pytorch: https://github.com/lucidrains/perceiver-pytorch
#
# ---- AROMA (https://github.com/LouisSerrano/aroma) ----
#
# MIT License
#
# Copyright (c) 2024 LouisSerrano
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---- perceiver-pytorch (https://github.com/lucidrains/perceiver-pytorch) ----
#
# MIT License
#
# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from einops import rearrange, repeat
from torch import nn

from ..inr.freq_embedding import (
    FourierPositionalEmbedding,
    MultiScaleNeRFEncoding,
    NeRFEncoding,
)
from .blocks import (
    Attention,
    CrossAttention,
    FeedForward,
    MultiScaleAttention,
    PreNorm,
    PreNormCross,
)
from .utils import DiagonalGaussianDistribution, cache_fn, exists

# ACKNOWLEDGEMENT: code adapted from Aroma implementation https://github.com/LouisSerrano/aroma,
# itself adapted from perceiver implementation by lucidrains


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 1,
        num_channels: int = 1,
        num_latents: int = 64,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        depth: int = 3,
        latent_heads: int = 8,
        latent_dim_head: int = 64,
        bottleneck_index: int = 0,
        cross_heads: int = 8,
        cross_dim_head: int = 64,
        max_pos_encoding_freq: int = 4,
        num_freq: int = 12,
        scales: Sequence[int] = [3, 4, 5],
        mlp_feature_dim: int = 16,
        decoder_ff: bool = False,
        encode_geo: bool = False,
        include_pos_in_value: bool = False,
        headwise_attn_output_gate: bool = False,
        use_kl: bool = True,
    ):
        """
        Modified AROMA encoder based on a Perceiver architecture.

        Output deep coordinate-based features for an INR decoder.

        Args:
            input_dim (int): number of coordinates
            num_channels (int): number of channels in values tensors
            num_latents (int): number of query tokens (*encode*)
            hidden_dim: (int): dim of the query tokens (*encode*)
            latent_dim: (int): dim of the tokens in latent space (*process*)
            depth (int): number of SA layers in latent space (*process*)
            latent_heads (int): number of latent self-attn heads (*process*)
            latent_dim_head (int): dim of each latent self-attn head (*process*)
            bottleneck_index (int): where to compress to `latent dim` wrt to the self-attn layers (*process*)
            cross_heads: (int): number of cross-attn heads (*encode* and *decode*)
            cross_dim_head (int): dim of each cross-attn head (*encode* and *decode*)
            max_pos_encoding_freq (int): max frequ (log2 scale) for Fourier feature pos embeddings
                of the geo encoder (*encode*)
            num_freq (int): number of frequ bands to sample for Fourier features (*encode* and *decode*)
            scales (Sequence[int]): in log2 scale, Fourier features bounds of each band (multi-band, *decode*)
            mlp_feature_dim: (int): feature dimensions before the MLP (*decode*)
            decoder_ff (bool): optional 2-layer MLP after multi-scale cross-attn (*decode*)
            encode_geo (bool): whether to encode the geomeotry (*encode*)
            include_pos_in_value (bool): whether to include coordinates in the values of the value (pixel)
                encoder (*encode*)
        """

        super().__init__()

        self.depth = depth
        self.bottleneck_index = (
            bottleneck_index  # where to put the botleneck, by default 0 means just after cross attention
        )
        self.encode_geo = encode_geo
        self.include_pos_in_value = include_pos_in_value
        self.use_kl = use_kl
        self.num_latents = num_latents

        # if include_pos_in_value, we must make sure that keys and values share the same dim
        # hence we call FourierPositionalEmbedding which NeRFEncoding + linear proj to hidden_dim
        # (values are lifted to hidden_dim)
        if include_pos_in_value:
            self.pos_encoding = FourierPositionalEmbedding(
                hidden_dim=hidden_dim,
                num_freq=num_freq,
                max_freq_log2=max_pos_encoding_freq,
                input_dim=input_dim,
                base_freq=2,
                use_relu=True,
            )
            key_dim = hidden_dim

        else:
            self.pos_encoding = NeRFEncoding(
                num_freq=num_freq,
                max_freq_log2=max_pos_encoding_freq,
                input_dim=input_dim,
                base_freq=2,
                log_sampling=True,
                include_input=True,
                min_freq_log2=0,
            )
            key_dim = self.pos_encoding.out_dim

        # initialize latent tokens:
        small_std = False
        sigma = 0.02 if small_std else 1
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim) * sigma)

        # get QK dim:
        value_dim = hidden_dim

        # a. project (scalar or 2D) values to higher dim space:
        self.lift_values = nn.Linear(num_channels, hidden_dim)

        # b. cross attend the coordinates:
        if self.encode_geo:
            self.cross_attend_geo = nn.ModuleList(
                [
                    PreNormCross(
                        hidden_dim,
                        CrossAttention(
                            query_dim=hidden_dim,
                            key_dim=key_dim,
                            value_dim=key_dim,
                            heads=cross_heads,
                            dim_head=cross_dim_head,
                            headwise_attn_output_gate=headwise_attn_output_gate,
                        ),
                        k_dim=key_dim,
                        v_dim=key_dim,
                    ),
                    PreNorm(hidden_dim, FeedForward(hidden_dim)),
                ]
            )

        # c. cross attend the pixels:
        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNormCross(
                    hidden_dim,
                    CrossAttention(
                        query_dim=hidden_dim,
                        key_dim=key_dim,
                        value_dim=value_dim,
                        heads=cross_heads,
                        dim_head=cross_dim_head,
                        headwise_attn_output_gate=headwise_attn_output_gate,
                    ),
                    k_dim=key_dim,
                    v_dim=value_dim,
                ),
                PreNorm(hidden_dim, FeedForward(hidden_dim)),
            ]
        )

        # d. self-attention in latent space:
        def get_latent_attn():
            return PreNorm(
                hidden_dim,
                Attention(
                    query_dim=hidden_dim,
                    heads=latent_heads,
                    dim_head=latent_dim_head,
                    headwise_attn_output_gate=headwise_attn_output_gate,
                ),
            )

        def get_latent_ff():
            return PreNorm(hidden_dim, FeedForward(hidden_dim))

        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers_before_mixing = nn.ModuleList([])
        self.layers_after_mixing = nn.ModuleList([])

        for _ in range(depth):
            self.layers_before_mixing.append(nn.ModuleList([get_latent_attn(), get_latent_ff()]))
            self.layers_after_mixing.append(nn.ModuleList([get_latent_attn(), get_latent_ff()]))

        # e. cross attend channels:
        def get_cross_attn():
            return PreNorm(
                hidden_dim,
                CrossAttention(
                    query_dim=hidden_dim,
                    key_dim=hidden_dim,
                    value_dim=hidden_dim,
                    heads=cross_heads,
                    dim_head=cross_dim_head,
                    headwise_attn_output_gate=headwise_attn_output_gate,
                ),
            )

        def get_cross_ff():
            return PreNorm(hidden_dim, FeedForward(hidden_dim))

        get_cross_attn, get_cross_ff = map(cache_fn, (get_cross_attn, get_cross_ff))
        self.cross_attend_channels = nn.ModuleList([])
        for _ in range(depth):
            self.cross_attend_channels.append(nn.ModuleList([get_cross_attn(), get_cross_ff()]))

        # f. multi-scale cross-attention decoder:
        self.pos_query = MultiScaleNeRFEncoding(
            num_freqs_per_scale=num_freq,
            log_sampling=True,
            include_input=True,
            input_dim=input_dim,
            base_freq=2,
            scales=scales,
            use_pi=True,
            disjoint=True,
        )
        queries_dim = self.pos_query.out_dim_per_scale

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            MultiScaleAttention(
                query_dim=queries_dim,
                context_dim=hidden_dim,
                out_dim=mlp_feature_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                headwise_attn_output_gate=headwise_attn_output_gate,
            ),
            context_dim=hidden_dim,
        )
        self.decoder_ff = PreNorm(mlp_feature_dim, FeedForward(mlp_feature_dim)) if decoder_ff else None

        self.mean_fc = nn.Linear(hidden_dim, latent_dim)
        if self.use_kl:
            self.logvar_fc = nn.Linear(hidden_dim, latent_dim)
        self.lift_z = nn.Linear(latent_dim, hidden_dim)
        self.out_dim = len(scales) * mlp_feature_dim

    def _pad(
        self,
        series: torch.Tensor,
        coords: torch.Tensor,
        series_covar: torch.Tensor,
        coords_covar: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:

        b = series.shape[0]
        missing_len = series_covar.shape[2] - series.shape[1]

        # handle nans in covar:
        mask_covar = series_covar.isnan().sum(0).squeeze(-1) == 0  # (C-1, covar_seq_len)
        series_covar = torch.nan_to_num(series_covar, nan=0.0)

        if missing_len > 0:
            mask = torch.cat(
                [
                    torch.cat([torch.ones(1, series.shape[1]), torch.zeros(1, missing_len)], dim=1).to(series.device),
                    mask_covar,
                    # torch.ones(series_covar.shape[1], series_covar.shape[2])
                ],
                dim=0,
            ).bool()  # (C, seq_len)
            # XXX merge with existing `mask`?

            series = torch.cat([series, torch.zeros((b, missing_len, 1)).to(series.device)], dim=1)  # (bs, seq_len, 1)

            coords = torch.cat([coords, torch.zeros((b, missing_len, 1)).to(series.device)], dim=1)  # (bs, seq_len, 1)

        elif missing_len < 0:
            mask = (
                torch.cat(
                    [
                        torch.ones(1, series.shape[1]).to(series.device),
                        torch.cat(
                            [
                                mask_covar,
                                # torch.ones(series_covar.shape[1], series_covar.shape[2]),
                                torch.zeros(series_covar.shape[1], abs(missing_len)).to(series.device),
                            ],
                            dim=1,
                        ),
                    ],
                    dim=0,
                )
                .bool()
                .to(series.device)
            )  # (C, seq_len)
            # XXX merge with existing `mask`?

            series_covar = torch.cat(
                [
                    series_covar,
                    torch.zeros((series.shape[0], coords_covar.shape[1], abs(missing_len), 1)).to(series.device),
                ],
                dim=2,
            )

            coords_covar = torch.cat(
                [
                    coords_covar,
                    torch.zeros((series.shape[0], coords_covar.shape[1], abs(missing_len), 1)).to(series.device),
                ],
                dim=2,
            )  # (bs, C-1, seq_len, 1)

        assert series.shape[1] == series_covar.shape[2] == coords.shape[1] == coords_covar.shape[2]

        return series, coords, series_covar, coords_covar, mask

    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor,
        series_covar: torch.Tensor | None = None,
        coords_covar: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        target_coords: torch.Tensor | None = None,
        sample_posterior: bool = True,
        return_stats: bool = False,
        return_latent_only: bool = False,
    ) -> (
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        series is shape (B, T_shared, 1)
        coords is shape (B, T_shared, 1)
        mask is shape (B, T_shared, 1)
        series_covar is shape (B, C-1, T_shared, 1)
        coords_covar is shape (B, C-1, T_shared, 1)
        """

        if series_covar is None:
            series_all = series.unsqueeze(1)  # (bs, 1, seq_len, 1)
            coords_all = coords.unsqueeze(1)  # (bs, 1, seq_len, 1)
        else:
            assert coords_covar is not None
            if coords_covar.ndim == 3:
                coords_covar = repeat(coords_covar, "b t 1 -> b c t 1", c=series_covar.shape[1])

            # pad to max_seq_len:
            series, coords, series_covar, coords_covar, mask = self._pad(
                series=series, coords=coords, series_covar=series_covar, coords_covar=coords_covar, mask=mask
            )

            series_all = torch.cat([series.unsqueeze(1), series_covar], dim=1)  # (bs, C, seq_len, 1)
            coords_all = torch.cat([coords.unsqueeze(1), coords_covar], dim=1)  # (bs, C, seq_len, 1)

            if mask is not None:
                assert mask.ndim == 2
                mask = repeat(mask, "c t -> (b c) t", b=series.shape[0])

        b, c, t, _ = series_all.shape

        series_all = rearrange(series_all, "b c t 1 -> (b c) t 1")  # (bs x C, seq_len, 1)
        coords_all = rearrange(coords_all, "b c t 1 -> (b c) t 1")  # (bs x C, seq_len, 1)

        # 1/5 Geometry encoder

        # prepare queries of the geo encoder:
        x = repeat(self.latents, "n d -> b n d", b=b * c)  # (bs x C, num_latents, hidden_dim)

        # prepare keys and values of the geo encoder:
        k = self.pos_encoding(coords_all)  # (bs x C, seq_len, )
        v = self.lift_values(series_all)  # (bs x C, seq_len, hidden_dim)

        # if encode_geo, cross attend to the other pixel locations
        if self.encode_geo:
            cross_attn, cross_ff = self.cross_attend_geo
            x = cross_attn(x, k=k, v=k, mask=mask) + x  # (bs x C, num_latents, hidden_dim)
            x = cross_ff(x) + x  # (bs x C, num_latents, hidden_dim)

        # 2/5 Value encoder

        # cross attend the coordinates:
        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(x, k=k, v=v + k if self.include_pos_in_value else v, mask=mask) + x
        )  # (bs x C, num_latents, hidden_dim)
        x = cross_ff(x) + x  # (bs x C, num_latents, hidden_dim)

        # 3/5 Process in latent space

        # L layers of self-attention on latent tokens:
        for index, (self_attn, self_ff) in enumerate(
            self.layers_before_mixing
        ):  # pyright: ignore[reportGeneralTypeIssues]
            if index == self.bottleneck_index:
                # bottleneck (compress hidden dim to latent dim)
                mu = self.mean_fc(x)  # (bs x C, num_latents, latent_dim)
                logvar = self.logvar_fc(x) if self.use_kl else mu  # (bs x C, num_latents, latent_dim)
                posterior = DiagonalGaussianDistribution(mu, logvar, deterministic=not self.use_kl)

                if sample_posterior:
                    z = posterior.sample()  # (bs x C, num_latents, latent_dim)
                else:
                    z = posterior.mode()  # (bs x C, num_latents, latent_dim)

                # back to hidden_dim:
                x = self.lift_z(z)  # (bs x C, num_latents, hidden_dim)

            # do self-attention (no context):
            x = self_attn(x) + x  # (bs x C, num_latents, hidden_dim)
            x = self_ff(x) + x  # (bs x C, num_latents, hidden_dim)

        # don't forget bottleneck if at the end of the self-attn layers:
        if self.bottleneck_index == len(self.layers_before_mixing):
            # bottleneck
            mu = self.mean_fc(x)
            logvar = self.logvar_fc(x) if self.use_kl else mu
            posterior = DiagonalGaussianDistribution(mu, logvar, deterministic=not self.use_kl)

            if self.use_kl and sample_posterior:
                z = posterior.sample()  # (bs x C, num_latents, latent_dim)
            else:
                z = posterior.mode()  # (bs x C, num_latents, latent_dim)

            # back to hidden_dim:
            x = self.lift_z(z)  # (bs x C, num_latents, hidden_dim)

        # 4/5 cross-attend the channels

        x = rearrange(x, "(b c) m d -> (b m) c d", b=b, c=c)  # (bs x num_latents, C, hidden_dim)

        # select target time series dimension:
        x_auto_reg = x[:, 0, :].unsqueeze(1)  # (bs x num_latents, 1, hidden_dim)

        # cross_attn, cross_ff = self.cross_attend_channels
        for cross_attn, cross_ff in self.cross_attend_channels:  # pyright: ignore[reportGeneralTypeIssues]
            # do cross-token attention:
            x_auto_reg = cross_attn(x_auto_reg, k=x, v=x, mask=None) + x_auto_reg  # (bs x num_latents, 1, hidden_dim)
            x_auto_reg = cross_ff(x_auto_reg) + x_auto_reg  # (bs x num_latents, 1, hidden_dim)

        # x = cross_attn(x_auto_reg, k=x, v=x, mask=None) + x_auto_reg    # (bs x num_latents, 1, hidden_dim)
        # x = cross_ff(x) + x                                             # (bs x num_latents, 1, hidden_dim)

        # rearrange and self attention block:
        # x = rearrange(x, '(b m) 1 d -> b m d', b=b, m=self.num_latents) # (bs, num_latents, hidden_dim)
        x = rearrange(x_auto_reg, "(b m) 1 d -> b m d", b=b, m=self.num_latents)  # (bs, num_latents, hidden_dim)

        for self_attn, self_ff in self.layers_after_mixing:  # pyright: ignore[reportGeneralTypeIssues]
            # do self-attention
            x = self_attn(x) + x  # (bs, num_latents, hidden_dim)
            x = self_ff(x) + x  # (bs, num_latents, hidden_dim)

        if return_latent_only:
            return x

        # 5/5 Cross-attn with target coords to get decoder features

        # get queries of the cross-attn decoder:
        if target_coords is None:
            queries = self.pos_query(coords)  # (bs, target_seq_len, num_bandwidth, pos_embed_dim)
        else:
            queries = self.pos_query(target_coords)  # (bs, target_seq_len, num_bandwidth, pos_embed_dim)

        # cross attend from decoder queries to latents:
        latents = self.decoder_cross_attn(queries, context=x)  # (bs, target_seq_len, num_bandwidth, mlp_feature_dim)

        # optional decoder feedforward:
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)  # (bs, target_seq_len, num_bandwidth, mlp_feature_dim)

        # get KL loss:
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if return_stats:
            return latents, kl_loss, mu, logvar

        return latents, kl_loss
