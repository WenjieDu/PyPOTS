"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.functional import nonstationary_norm, nonstationary_denorm
from ...nn.modules.loss import Criterion
from ...nn.modules.timesnet import BackboneTimesNet
from ...nn.modules.transformer.embedding import DataEmbedding


class _TimesNet(nn.Module):
    def __init__(
        self,
        n_layers,
        n_steps,
        n_features,
        n_pred_steps,
        n_pred_features,
        top_k,
        d_model,
        d_ffn,
        n_kernels,
        dropout,
        apply_nonstationary_norm,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.pred_len = n_pred_steps
        self.n_pred_features = n_pred_features
        self.n_layers = n_layers
        self.apply_nonstationary_norm = apply_nonstationary_norm
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.enc_embedding = DataEmbedding(
            n_features,
            d_model,
            dropout=dropout,
            n_max_steps=n_steps,
        )
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.model = BackboneTimesNet(
            n_layers,
            n_steps,
            n_pred_steps,
            top_k,
            d_model,
            d_ffn,
            n_kernels,
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.projection = nn.Linear(d_model, n_pred_features)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)

        # embedding
        enc_out = self.enc_embedding(X)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        # TimesNet processing
        enc_out = self.model(enc_out)

        # project back the original data space
        dec_out = self.projection(enc_out)

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = nonstationary_denorm(dec_out, means, stdev)

        forecasting_result = dec_out[:, -self.pred_len :]

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
