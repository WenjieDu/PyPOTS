"""
The core wrapper assembles the submodules of TSLANet imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.modules import ModelCore
from ...nn.modules.tslanet import BackboneTSLANet


class _TSLANet(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        patch_size: int,
        d_embedding: int,
        dropout: float,
        mask_ratio: float,
    ):
        super().__init__()

        self.n_steps = n_steps

        self.backbone = BackboneTSLANet(
            "imputation",
            n_steps,
            n_features,
            n_steps,
            n_layers,
            patch_size,
            d_embedding,
            dropout,
            mask_ratio,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        results = {}
        if calc_criterion:
            reconstruction, target, mask = self.backbone.pretrain(X)
            loss = (reconstruction - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = loss

        else:
            reconstruction = self.backbone(X)
            imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

            results["imputation"] = imputed_data
            results["reconstruction"] = reconstruction

        return results
