"""
The core wrapper assembles the submodules of ModernTCN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.functional import nonstationary_norm, nonstationary_denorm
from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.moderntcn import BackboneModernTCN
from ...nn.modules.patchtst.layers import FlattenHead


class _ModernTCN(ModelCore):
    def __init__(
        self,
        n_steps,
        n_features,
        patch_size,
        patch_stride,
        downsampling_ratio,
        ffn_ratio,
        num_blocks: list,
        large_size: list,
        small_size: list,
        dims: list,
        small_kernel_merged: bool,
        backbone_dropout: float,
        head_dropout: float,
        use_multi_scale: bool,
        individual: bool,
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

        self.backbone = BackboneModernTCN(
            n_steps,
            n_features,
            n_features,
            patch_size,
            patch_stride,
            downsampling_ratio,
            ffn_ratio,
            num_blocks,
            large_size,
            small_size,
            dims,
            small_kernel_merged,
            backbone_dropout,
            head_dropout,
            use_multi_scale,
            individual,
        )

        # for the imputation task, the output dim is the same as input dim
        self.projection = FlattenHead(
            self.backbone.head_nf,
            n_steps,
            n_features,
            head_dropout,
            individual,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)

        in_X = X.permute(0, 2, 1)
        in_X = self.backbone(in_X)
        reconstruction = self.projection(in_X)
        reconstruction = reconstruction.permute(0, 2, 1)

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            reconstruction = nonstationary_denorm(reconstruction, means, stdev)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                loss = self.training_loss(reconstruction, X, missing_mask)
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
