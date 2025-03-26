"""
The core wrapper assembles the submodules of GPT4TS imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.modules import ModelCore
from ...nn.modules.gpt4ts import BackboneGPT4TS
from ...nn.modules.loss import Criterion


class _GPT4TS(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        patch_size: int,
        patch_stride: int,
        train_gpt_mlp: bool,
        d_ffn: int,
        dropout: float,
        embed: str,
        freq: str,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.backbone = BackboneGPT4TS(
            "imputation",
            n_steps,
            n_features,
            0,
            n_features,
            n_layers,
            patch_size,
            patch_stride,
            train_gpt_mlp,
            d_ffn,
            dropout,
            embed,
            freq,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # GPT4TS backbone processing
        reconstruction = self.backbone(X, mask=missing_mask)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                loss = self.training_loss(reconstruction, X_ori, indicating_mask)
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
