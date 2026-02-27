"""
The core wrapper assembles the submodules of TimeMixer++ forecasting model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.timemixerpp import BackboneTimeMixerPP


class _TimeMixerPP(ModelCore):
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
        n_heads: int,
        dropout: float,
        top_k: int,
        n_kernels: int,
        channel_mixing: bool,
        channel_independence: bool,
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
        self.model = BackboneTimeMixerPP(
            task_name=term + "_term_forecast",
            n_steps=n_steps,
            n_features=n_features,
            n_pred_steps=n_pred_steps,
            n_pred_features=n_pred_features,
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            n_heads=n_heads,
            dropout=dropout,
            top_k=top_k,
            n_kernels=n_kernels,
            channel_mixing=channel_mixing,
            channel_independence=channel_independence,
            downsampling_layers=downsampling_layers,
            downsampling_window=downsampling_window,
            downsampling_method="avg",
            use_future_temporal_feature=False,
            use_norm=use_norm,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X = inputs["X"]

        # TimeMixer++ forecasting
        forecasting_result = self.model.forecast(X, None)

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
