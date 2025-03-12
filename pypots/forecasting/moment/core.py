"""
The core wrapper assembles the submodules of MOMENT forecasting model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules.loss import Criterion, MSE
from ...nn.modules.moment import BackboneMOMENT, SUPPORTED_HUGGINGFACE_MODELS
from ...nn.modules.saits import SaitsEmbedding


class _MOMENT(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_pred_features: int,
        term: str,
        transformer_backbone: str,
        transformer_type: str,
        patch_size: int,
        patch_stride: int,
        d_model: int,
        d_ffn: int,
        dropout: float,
        head_dropout: float,
        finetuning_mode: str,
        revin_affine: bool = False,
        add_positional_embedding: bool = False,
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
        mask_ratio: float = 0,
        device: str = "cpu",
        training_loss: Criterion = MSE(),
    ):
        super().__init__()

        assert term in ["long", "short"], "forecasting term should be either 'long' or 'short'"

        self.n_steps = n_steps
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.training_loss = training_loss

        configs = {
            "task_name": term + "_term_forecasting",
            "seq_len": n_steps,
            "forecast_horizon": n_pred_steps,
            "patch_len": patch_size,
            "patch_stride_len": patch_stride,
            "revin_affine": revin_affine,
            "d_model": d_model,
            "d_ff": d_ffn,
            "dropout": dropout,
            "head_dropout": head_dropout,
            "add_positional_embedding": add_positional_embedding,
            "value_embedding_bias": value_embedding_bias,
            "orth_gain": orth_gain,
            "mask_ratio": mask_ratio,
            "device": device,
            "transformer_backbone": transformer_backbone,
            "transformer_type": transformer_type,
            "finetuning_mode": finetuning_mode,
        }
        self.backbone = BackboneMOMENT(configs)

        if finetuning_mode == "linear-probing":
            for name, param in self.backbone.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False

        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # WDU: the original PatchTST paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        input_X = self.saits_embedding(X, missing_mask)
        batch_size = X.shape[0]
        input_mask = torch.ones([batch_size, self.n_steps + self.n_pred_steps], device=X.device)
        input_mask[:, self.n_steps :] = 0
        input_X = nn.functional.pad(input_X, (0, 0, 0, self.n_pred_steps), "constant", 0)
        # missing_mask = nn.functional.pad(missing_mask, (0, 0, 0, self.n_pred_steps), "constant", 0)
        input_X = input_X.permute(0, 2, 1)

        # MOMENT backbone processing
        forecasting_result = self.backbone(input_X, input_mask=input_mask, forecast_horizon=self.n_pred_steps).forecast
        forecasting_result = forecasting_result.permute(0, 2, 1)
        forecasting_result = forecasting_result[:, :, : self.n_pred_features]

        results = {
            "forecasting_data": forecasting_result,
        }

        # if in training mode, return results with losses
        if self.training:
            X_pred, X_pred_missing_mask = inputs["X_pred"], inputs["X_pred_missing_mask"]
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = self.training_loss(X_pred, forecasting_result, X_pred_missing_mask)

        return results
