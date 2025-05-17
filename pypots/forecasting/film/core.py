"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.film import BackboneFiLM
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import SaitsEmbedding


class _FiLM(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_pred_steps,
        n_pred_features,
        window_size,
        multiscale,
        modes1,
        ratio,
        mode_type,
        d_model,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_pred_features = n_pred_features

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=False,
        )
        self.backbone = BackboneFiLM(
            n_steps,
            d_model,
            n_pred_steps,
            window_size,
            multiscale,
            modes1,
            ratio,
            mode_type,
        )
        self.output_projection = nn.Linear(d_model, n_pred_features)

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

        X_embedding = self.saits_embedding(X, missing_mask)

        # FiLM processing
        backbone_output = self.backbone(X_embedding)
        forecasting_result = self.output_projection(backbone_output)
        forecasting_result = forecasting_result[:, :, : self.n_pred_features]  # select the first n_pred_features

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
