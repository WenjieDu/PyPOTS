"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.functional import calc_mse
from ...nn.functional import (
    nonstationary_norm,
    nonstationary_denorm,
)
from ...nn.modules.timemixer import BackboneTimeMixer


class _TimeMixer(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        dropout: float,
        top_k: int,
        channel_independence: bool,
        decomp_method: str,
        moving_avg: int,
        downsampling_layers: int,
        downsampling_window: int,
        apply_nonstationary_norm: bool = False,
    ):
        super().__init__()

        self.apply_nonstationary_norm = apply_nonstationary_norm

        self.model = BackboneTimeMixer(
            task_name="imputation",
            n_steps=n_steps,
            n_features=n_features,
            n_pred_steps=0,
            n_pred_features=n_features,
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            channel_independence=channel_independence,
            decomp_method=decomp_method,
            top_k=top_k,
            moving_avg=moving_avg,
            downsampling_layers=downsampling_layers,
            downsampling_window=downsampling_window,
            downsampling_method="avg",
            use_future_temporal_feature=False,
        )

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)

        # TimesMixer processing
        dec_out = self.model.imputation(X, None)

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = nonstationary_denorm(dec_out, means, stdev)

        imputed_data = missing_mask * X + (1 - missing_mask) * dec_out
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if self.training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(dec_out, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
