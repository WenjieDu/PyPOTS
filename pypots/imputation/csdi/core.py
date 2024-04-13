# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.csdi import BackboneCSDI


class _CSDI(nn.Module):
    def __init__(
        self,
        n_layers,
        n_heads,
        n_channels,
        d_target,
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

        self.backbone = BackboneCSDI(
            n_layers,
            n_heads,
            n_channels,
            d_target,
            d_time_embedding,
            d_feature_embedding,
            d_diffusion_embedding,
            is_unconditional,
            n_diffusion_steps,
            schedule,
            beta_start,
            beta_end,
        )

    def forward(self, inputs, training=True, n_sampling_times=1):
        results = {}
        if training:  # for training
            (observed_data, indicating_mask, cond_mask, observed_tp) = (
                inputs["X_ori"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            side_info = self.backbone.get_side_info(observed_tp, cond_mask)
            training_loss = self.backbone.calc_loss(
                observed_data, cond_mask, indicating_mask, side_info, training
            )
            results["loss"] = training_loss
        elif not training and n_sampling_times == 0:  # for validating
            (observed_data, indicating_mask, cond_mask, observed_tp) = (
                inputs["X_ori"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            side_info = self.backbone.get_side_info(observed_tp, cond_mask)
            validating_loss = self.backbone.calc_loss_valid(
                observed_data, cond_mask, indicating_mask, side_info, training
            )
            results["loss"] = validating_loss
        elif not training and n_sampling_times > 0:  # for testing
            observed_data, cond_mask, observed_tp = (
                inputs["X"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            side_info = self.backbone.get_side_info(observed_tp, cond_mask)
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
