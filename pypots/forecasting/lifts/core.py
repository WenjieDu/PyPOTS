"""LIFTS forecasting core."""

# Created by Sanyou Wu <sw4180@cumc.columbia.edu>
# License: BSD-3-Clause

from ...nn.modules import ModelCore
from ...nn.modules.lifts import BackboneLIFTS
from ...nn.modules.lifts.training import reference_batch_losses


class _LIFTS(ModelCore):
    def __init__(
        self,
        n_pred_steps,
        n_steps,
        n_features,
        d_latent,
        d_model,
        n_heads,
        d_ffn,
        n_layers,
        n_inner_layers,
        self_mask_rate,
        self_mask_weight,
        mask_weight,
        n_sampling_times_train,
        validation_metric,
    ):
        super().__init__()
        self.backbone = BackboneLIFTS(
            n_steps,
            n_features,
            d_latent,
            d_model,
            n_heads,
            d_ffn,
            n_layers,
            n_inner_layers,
        )
        self.self_mask_rate = self_mask_rate
        self.self_mask_weight = self_mask_weight
        self.mask_weight = mask_weight
        self.n_sampling_times_train = n_sampling_times_train
        self.validation_metric = validation_metric
        self.n_pred_steps = n_pred_steps

    def forward(self, inputs, calc_criterion=False, n_sampling_times=100):
        if not isinstance(n_sampling_times, int) or isinstance(n_sampling_times, bool) or n_sampling_times < 1:
            raise ValueError("n_sampling_times must be a positive integer.")
        X, mask = inputs["X"], inputs["missing_mask"]
        if self.training and calc_criterion:
            # The standalone objective trains on the observed history only.
            # Future targets must not change its training sequence or RNG flow.
            return reference_batch_losses(self, inputs)
        samples = self.backbone.forecast(X, mask, self.n_pred_steps, n_sampling_times)
        results = {"forecasting": samples.mean(dim=1), "forecasting_samples": samples}
        if calc_criterion:
            if not inputs["X_pred_missing_mask"].any():
                raise ValueError("Validation X_pred must contain at least one observed target.")
            results["metric"] = self.validation_metric(
                results["forecasting"],
                inputs["X_pred"],
                inputs["X_pred_missing_mask"],
            )
        return results
