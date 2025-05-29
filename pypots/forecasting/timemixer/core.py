"""
The core wrapper assembles the submodules of TimeMixer forecasting model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.timemixer import BackboneTimeMixer


class _TimeMixer(ModelCore):
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
        use_norm: bool,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

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

        self.output_projection = nn.Linear(n_features, n_pred_features)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X = inputs["X"]
        # missing_mask = inputs["missing_mask"]

        # TimesMixer processing
        enc_out = self.model.forecast(X, None)

        # project back the original data space
        forecasting_result = self.output_projection(enc_out)
        # the raw output has length = n_steps+n_pred_steps, we only need the last n_pred_steps
        forecasting_result = forecasting_result[:, -self.n_pred_steps :]

        results = {
            "forecasting": forecasting_result,
        }

        if calc_criterion:
            X_pred, X_pred_missing_mask = inputs["X_pred"], inputs["X_pred_missing_mask"]
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = self.training_loss(X_pred, forecasting_result, X_pred_missing_mask)
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = self.validation_metric(X_pred, forecasting_result, X_pred_missing_mask)

        return results
