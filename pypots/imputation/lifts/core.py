"""LIFTS imputation core."""

from ...nn.modules import ModelCore
from ...nn.modules.lifts import BackboneLIFTS


class _LIFTS(ModelCore):
    def __init__(
        self,
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

    def forward(self, inputs, calc_criterion=False, n_sampling_times=100):
        if not isinstance(n_sampling_times, int) or isinstance(n_sampling_times, bool) or n_sampling_times < 1:
            raise ValueError("n_sampling_times must be a positive integer.")
        X, mask = inputs["X"], inputs["missing_mask"]
        if self.training and calc_criterion:
            loss, forecast_loss, self_mask_loss = self.backbone.joint_loss(
                X,
                mask,
                self.self_mask_rate,
                self.self_mask_weight,
                self.mask_weight,
                self.n_sampling_times_train,
            )
            return {"loss": loss, "forecast_loss": forecast_loss, "self_mask_loss": self_mask_loss}
        samples = self.backbone.impute(X, mask, n_sampling_times)
        results = {"imputation": samples.mean(dim=1), "imputation_samples": samples}
        if calc_criterion:
            if not inputs["indicating_mask"].any():
                raise ValueError("Validation requires held-out observed values in X_ori that are missing in X.")
            results["metric"] = self.validation_metric(
                results["imputation"],
                inputs["X_ori"],
                inputs["indicating_mask"],
            )
        return results
