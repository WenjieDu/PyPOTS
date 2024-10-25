"""
The core wrapper assembles the submodules of SegRNN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Shengsheng Lin

from typing import Optional

from typing import Callable
import torch.nn as nn

from ...nn.modules.segrnn import BackboneSegRNN
from ...nn.modules.saits import SaitsLoss


class _SegRNN(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        seg_len: int = 24,
        d_model: int = 512,
        dropout: float = 0.5,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.seg_len = seg_len
        self.d_model = d_model
        self.dropout = dropout

        self.backbone = BackboneSegRNN(n_steps, n_features, seg_len, d_model, dropout)

        # apply SAITS loss function to Transformer on the imputation task
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        reconstruction = self.backbone(X)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss

        return results
