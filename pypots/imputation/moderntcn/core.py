"""
The core wrapper assembles the submodules of ModernTCN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.functional import nonstationary_norm, nonstationary_denorm
from ...nn.functional import calc_mse
from ...nn.modules.moderntcn import BackboneModernTCN
from ...nn.modules.patchtst.layers import FlattenHead


class _ModernTCN(nn.Module):
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
        small_kernel_merged: bool = False,
        backbone_dropout: float = 0.1,
        head_dropout: float = 0.1,
        use_multi_scale: bool = True,
        individual: bool = False,
        apply_nonstationary_norm: bool = False,
    ):
        super().__init__()

        self.apply_nonstationary_norm = apply_nonstationary_norm

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

    def forward(self, inputs: dict) -> dict:
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
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if self.training:
            loss = calc_mse(reconstruction, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
