"""
The core wrapper assembles the submodules of TimeMixer forecasting model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.functional.error import calc_mse
from ...nn.modules.timemixer import BackboneTimeMixer


class _TimeMixer(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_pred_features: int,
        term: str,
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
        use_norm: bool = False,
    ):
        super().__init__()

        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features

        assert term in ["long", "short"], "forecasting term should be either 'long' or 'short'"
        self.model = BackboneTimeMixer(
            task_name=term + "_term_forecast",
            n_steps=n_steps,
            n_features=n_features,
            n_pred_steps=n_pred_steps,
            n_pred_features=n_pred_features,
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
            use_norm=use_norm,
        )

        # for the imputation task, the output dim is the same as input dim
        self.output_projection = nn.Linear(n_features, n_pred_features)

    def forward(self, inputs: dict) -> dict:
        X = inputs["X"]
        # missing_mask = inputs["missing_mask"]

        if self.training:
            X_pred, X_pred_missing_mask = inputs["X_pred"], inputs["X_pred_missing_mask"]
        else:
            batch_size = X.shape[0]
            X_pred, X_pred_missing_mask = (
                torch.zeros(batch_size, self.n_pred_steps, self.n_pred_features),
                torch.ones(batch_size, self.n_pred_steps, self.n_pred_features),
            )

        # TimesMixer processing
        # WDU: missing_mask should not be passed into the model forward processing because the official implementation
        # does not accept POTS on the forecasting task. And if pass in, it will result in
        # x and x_mark shape not consistent bug
        # enc_out = self.model.forecast(X, missing_mask)
        enc_out = self.model.forecast(X, None)

        # project back the original data space
        forecasting_result = self.output_projection(enc_out)
        # the raw output has length = n_steps+n_pred_steps, we only need the last n_pred_steps
        forecasting_result = forecasting_result[:, -self.n_pred_steps :]

        results = {
            "forecasting_data": forecasting_result,
        }

        # if in training mode, return results with losses
        if self.training:
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = calc_mse(X_pred, forecasting_result, X_pred_missing_mask)

        return results
