"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.autoformer import SeriesDecompositionBlock
from ...nn.modules.dlinear import BackboneDLinear
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import SaitsEmbedding


class _DLinear(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_pred_steps,
        n_pred_features,
        moving_avg_window_size: int,
        individual: bool,
        d_model: int,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.pred_len = n_pred_steps
        self.n_pred_features = n_pred_features

        self.individual = individual

        self.series_decomp = SeriesDecompositionBlock(moving_avg_window_size)
        self.backbone = BackboneDLinear(n_steps, n_features, individual, d_model)

        if not individual:
            self.seasonal_saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
            self.trend_saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
            self.linear_seasonal_output = nn.Linear(d_model, n_features)
            self.linear_trend_output = nn.Linear(d_model, n_features)

        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # input preprocessing and embedding for DLinear
        seasonal_init, trend_init = self.series_decomp(X)

        if not self.individual:
            seasonal_init = self.seasonal_saits_embedding(seasonal_init, missing_mask)
            trend_init = self.trend_saits_embedding(trend_init, missing_mask)

        seasonal_output, trend_output = self.backbone(seasonal_init, trend_init)

        if not self.individual:
            seasonal_output = self.linear_seasonal_output(seasonal_output)
            trend_output = self.linear_trend_output(trend_output)

        output = seasonal_output + trend_output

        forecasting_result = output[:, -self.pred_len :]

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
