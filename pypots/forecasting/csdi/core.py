"""
The core wrapper assembles the submodules of CSDI forecasting model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.csdi import BackboneCSDI


class _CSDI(ModelCore):
    def __init__(
        self,
        n_features,
        n_pred_features,
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
        self.n_pred_features = n_pred_features
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
            n_pred_features,
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

    def get_side_info(self, observed_tp, cond_mask, feature_id):
        B, K, L = cond_mask.shape
        device = observed_tp.device
        time_embed = self.time_embedding(observed_tp, self.d_time_embedding)  # (B,L,emb)
        time_embed = time_embed.to(device)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.n_pred_features, -1)

        if self.n_pred_features == self.n_features:
            feature_embed = self.embed_layer(torch.arange(self.n_pred_features).to(device))  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,emb+d_feature_embedding)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(
        self,
        inputs,
        calc_criterion: bool = False,
        n_sampling_times=1,
    ) -> dict:
        results = {}

        if calc_criterion:
            if self.training:  # for training
                (observed_data, indicating_mask, cond_mask, observed_tp, feature_id) = (
                    inputs["X_ori"],
                    inputs["indicating_mask"],
                    inputs["cond_mask"],
                    inputs["observed_tp"],
                    inputs["feature_id"],
                )
                side_info = self.get_side_info(observed_tp, cond_mask, feature_id)
                training_loss = self.backbone.calc_loss(observed_data, cond_mask, indicating_mask, side_info)
                results["loss"] = training_loss
            else:  # for validating
                (observed_data, indicating_mask, cond_mask, observed_tp, feature_id) = (
                    inputs["X_ori"],
                    inputs["indicating_mask"],
                    inputs["cond_mask"],
                    inputs["observed_tp"],
                    inputs["feature_id"],
                )
                side_info = self.get_side_info(observed_tp, cond_mask, feature_id)
                validating_loss = self.backbone.calc_loss_valid(observed_data, cond_mask, indicating_mask, side_info)
                results["metric"] = validating_loss
        else:
            observed_data, cond_mask, observed_tp, feature_id = (
                inputs["X"],
                inputs["cond_mask"],
                inputs["observed_tp"],
                inputs["feature_id"],
            )
            side_info = self.get_side_info(observed_tp, cond_mask, feature_id)
            samples = self.backbone(
                observed_data, cond_mask, side_info, n_sampling_times
            )  # (n_samples, n_sampling_times, n_features, n_steps)
            repeated_obs = observed_data.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            repeated_mask = cond_mask.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            forecasting = repeated_obs + samples * (1 - repeated_mask)

            results["forecasting"] = forecasting.permute(
                0, 1, 3, 2
            )  # (n_samples, n_sampling_times, n_steps, n_features)

        return results
