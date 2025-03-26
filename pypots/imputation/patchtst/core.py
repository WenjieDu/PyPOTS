"""
The core wrapper assembles the submodules of PatchTST imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.patchtst import PatchEmbedding, PatchtstEncoder, PredictionHead
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding


class _PatchTST(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        patch_size: int,
        patch_stride: int,
        dropout: float,
        attn_dropout: float,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        n_patches = int((n_steps - patch_size) / patch_stride + 2)  # number of patches
        padding = patch_stride

        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
        self.patch_embedding = PatchEmbedding(d_model, patch_size, patch_stride, padding, dropout)
        self.encoder = PatchtstEncoder(
            n_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
        )
        self.head = PredictionHead(d_model, n_patches, n_steps, dropout)
        self.output_projection = nn.Linear(d_model, n_features)
        # apply SAITS loss function to PatchTST on the imputation task
        self.training_loss = SaitsLoss(ORT_weight, MIT_weight, training_loss)
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # WDU: the original PatchTST paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        input_X = self.saits_embedding(X, missing_mask)

        # do patch  embedding
        enc_out = self.patch_embedding(input_X.permute(0, 2, 1))  # [bz * d_model, n_patches, d_model]

        # PatchTST encoder processing
        enc_out, attns = self.encoder(enc_out)

        # project back the original data space
        dec_out = self.head(enc_out)  # [bz, n_steps, d_model]
        reconstruction = self.output_projection(dec_out)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                loss, ORT_loss, MIT_loss = self.training_loss(reconstruction, X_ori, missing_mask, indicating_mask)
                results["ORT_loss"] = ORT_loss
                results["MIT_loss"] = MIT_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
