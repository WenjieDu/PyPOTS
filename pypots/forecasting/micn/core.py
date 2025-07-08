"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.fedformer.layers import SeriesDecompositionMultiBlock
from ...nn.modules.loss import Criterion
from ...nn.modules.micn import BackboneMICN
from ...nn.modules.saits import SaitsEmbedding


class _MICN(nn.Module):
    def __init__(
        self,
        n_layers,
        n_steps,
        n_features,
        n_pred_steps,
        n_pred_features,
        d_model: int,
        dropout: float,
        conv_kernel: list,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.pred_len = n_pred_steps
        self.n_pred_features = n_pred_features
        self.n_layers = n_layers
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=True,
            dropout=dropout,
        )

        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((n_steps + n_steps + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((n_steps + n_steps + ii - 1) // ii)

        self.decomp_multi = SeriesDecompositionMultiBlock(decomp_kernel)
        self.backbone = BackboneMICN(
            n_steps,
            n_features,
            n_pred_steps,
            n_pred_features,
            n_layers,
            d_model,
            decomp_kernel,
            isometric_kernel,
            conv_kernel,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        seasonal_init, trend_init = self.decomp_multi(X)

        enc_out = self.saits_embedding(seasonal_init, missing_mask)

        # MICN encoder processing
        enc_out = self.backbone(enc_out)
        enc_out = enc_out + trend_init

        forecasting_result = enc_out[:, -self.pred_len :]

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
