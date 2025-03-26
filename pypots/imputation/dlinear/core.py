"""
The core wrapper assembles the submodules of DLinear imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional

import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.autoformer import SeriesDecompositionBlock
from ...nn.modules.dlinear import BackboneDLinear
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding


class _DLinear(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        moving_avg_window_size: int,
        individual: bool,
        d_model: Optional[int],
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.individual = individual

        self.series_decomp = SeriesDecompositionBlock(moving_avg_window_size)
        self.backbone = BackboneDLinear(n_steps, n_features, individual, d_model)

        if not individual:
            self.seasonal_saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
            self.trend_saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
            self.linear_seasonal_output = nn.Linear(d_model, n_features)
            self.linear_trend_output = nn.Linear(d_model, n_features)

        # apply SAITS loss function to DLinear on the imputation task
        self.training_loss = SaitsLoss(ORT_weight, MIT_weight, training_loss)
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
            # WDU: the original DLinear paper isn't proposed for imputation task. Hence the model doesn't take
            # the missing mask into account, which means, in the process, the model doesn't know which part of
            # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
            # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
            # the output layers to project the seasonal and trend from the hidden space to the original space.
            # But this is only for the non-individual mode.
            seasonal_init = self.seasonal_saits_embedding(seasonal_init, missing_mask)
            trend_init = self.trend_saits_embedding(trend_init, missing_mask)

        seasonal_output, trend_output = self.backbone(seasonal_init, trend_init)

        if not self.individual:
            seasonal_output = self.linear_seasonal_output(seasonal_output)
            trend_output = self.linear_trend_output(trend_output)

        reconstruction = seasonal_output + trend_output

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                loss, ORT_loss, MIT_loss = self.training_loss(reconstruction, X_ori, missing_mask, indicating_mask)
                results["ORT_loss"] = ORT_loss
                results["MIT_loss"] = MIT_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
