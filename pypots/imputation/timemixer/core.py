"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.functional import (
    nonstationary_norm,
    nonstationary_denorm,
)
from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.timemixer import BackboneTimeMixer


class _TimeMixer(ModelCore):
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
        apply_nonstationary_norm: bool,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.apply_nonstationary_norm = apply_nonstationary_norm
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

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
        reconstruction = self.model.imputation(X, None)

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            reconstruction = nonstationary_denorm(reconstruction, means, stdev)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
            "reconstruction": reconstruction,
        }

        return results

    def calc_criterion(self, inputs: dict) -> dict:
        results = self.forward(inputs)
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        reconstruction = results["reconstruction"]

        if self.training:  # if in the training mode (the training stage), return loss result from training_loss
            # `loss` is always the item for backward propagating to update the model
            loss = self.training_loss(reconstruction, X, missing_mask)
            results["loss"] = loss
        else:  # if in the eval mode (the validation stage), return metric result from validation_metric
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
