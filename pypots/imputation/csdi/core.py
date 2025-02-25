"""
The core wrapper assembles the submodules of CSDI imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules.csdi import BackboneCSDI


class _CSDI(nn.Module):
    def __init__(
        self,
        n_features,
        n_layers,
        n_heads,
        n_channels,
        d_time_embedding,
        d_feature_embedding,
        d_diffusion_embedding,
        is_unconditional,
        n_diffusion_steps,
        schedule,
        beta_start,
        beta_end,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_time_embedding = d_time_embedding
        self.is_unconditional = is_unconditional

        self.embed_layer = nn.Embedding(
            num_embeddings=n_features,
            embedding_dim=d_feature_embedding,
        )
        self.backbone = BackboneCSDI(
            n_layers,
            n_heads,
            n_channels,
            n_features,
            d_time_embedding,
            d_feature_embedding,
            d_diffusion_embedding,
            is_unconditional,
            n_diffusion_steps,
            schedule,
            beta_start,
            beta_end,
        )

    @staticmethod
    def time_embedding(pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(pos.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=pos.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        device = observed_tp.device
        time_embed = self.time_embedding(observed_tp, self.d_time_embedding)  # (B,L,emb)
        time_embed = time_embed.to(device)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(self.n_features).to(device))  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,emb+d_feature_embedding)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, inputs, n_sampling_times=1):
        results = {}
        if self.training:  # for training
            (observed_data, indicating_mask, cond_mask, observed_tp) = (
                inputs["X_ori"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            side_info = self.get_side_info(observed_tp, cond_mask)
            training_loss = self.backbone.calc_loss(observed_data, cond_mask, indicating_mask, side_info)
            results["loss"] = training_loss
        elif not self.training and n_sampling_times == 0:  # for validating
            (observed_data, indicating_mask, cond_mask, observed_tp) = (
                inputs["X_ori"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            side_info = self.get_side_info(observed_tp, cond_mask)
            validating_loss = self.backbone.calc_loss_valid(observed_data, cond_mask, indicating_mask, side_info)
            results["loss"] = validating_loss
        elif not self.training and n_sampling_times > 0:  # for testing
            observed_data, cond_mask, observed_tp = (
                inputs["X"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.backbone(
                observed_data, cond_mask, side_info, n_sampling_times
            )  # (n_samples, n_sampling_times, n_features, n_steps)
            repeated_obs = observed_data.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            repeated_mask = cond_mask.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            imputed_data = repeated_obs + samples * (1 - repeated_mask)

            results["imputed_data"] = imputed_data.permute(
                0, 1, 3, 2
            )  # (n_samples, n_sampling_times, n_steps, n_features)

        return results
