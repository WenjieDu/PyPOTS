"""
The core wrapper assembles the submodules of TEFN forecasting model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.functional import nonstationary_norm, nonstationary_denorm
from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import SaitsEmbedding
from ...nn.modules.tefn import BackboneTEFN


class _TEFN(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_pred_features: int,
        n_fod: int,
        apply_nonstationary_norm: bool,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.apply_nonstationary_norm = apply_nonstationary_norm
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.saits_embedding = SaitsEmbedding(
            n_steps * 2,
            n_steps + n_pred_steps,
            with_pos=False,
        )
        self.backbone = BackboneTEFN(
            n_steps,
            n_features,
            n_pred_steps,
            n_fod,
        )

        # for the imputation task, the output dim is the same as input dim
        self.output_projection = nn.Linear(n_features, n_pred_features)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)

        # WDU: the original TEFN paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        enc_out = self.saits_embedding(X.permute(0, 2, 1), missing_mask.permute(0, 2, 1)).permute(0, 2, 1)

        # TEFN encoder processing
        enc_out = self.backbone(enc_out)
        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            enc_out = nonstationary_denorm(enc_out, means, stdev)

        # project back the original data space
        forecasting_result = self.output_projection(enc_out)
        # the raw output has length = n_steps+n_pred_steps, we only need the last n_pred_steps
        forecasting_result = forecasting_result[:, -self.n_pred_steps :]

        results = {
            "forecasting_result": forecasting_result,
        }

        return results

    def calc_criterion(self, inputs: dict) -> dict:
        results = self.forward(inputs)

        X_pred, X_pred_missing_mask = inputs["X_pred"], inputs["X_pred_missing_mask"]
        forecasting_result = results["forecasting_result"]

        if self.training:  # if in the training mode (the training stage), return loss result from training_loss
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = self.training_loss(X_pred, forecasting_result, X_pred_missing_mask)
        else:  # if in the eval mode (the validation stage), return metric result from validation_metric
            results["metric"] = self.validation_metric(X_pred, forecasting_result, X_pred_missing_mask)

        return results
