"""
The core wrapper assembles the submodules of MOMENT imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.moment import BackboneMOMENT
from ...nn.modules.saits import SaitsEmbedding


class _MOMENT(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        transformer_backbone: str,
        transformer_type: str,
        patch_size: int,
        patch_stride: int,
        d_model: int,
        d_ffn: int,
        dropout: float,
        head_dropout: float,
        finetuning_mode: str,
        revin_affine: bool,
        add_positional_embedding: bool,
        value_embedding_bias: bool,
        orth_gain: float,
        mask_ratio: float,
        device: str,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        configs = {
            "task_name": "pre-training",
            "seq_len": n_steps,
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
            "debug": True,
        }
        self.backbone = BackboneMOMENT(configs)

        if finetuning_mode == "linear-probing":
            for name, param in self.backbone.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False

        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)

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
        batch_size = X.shape[0]
        input_mask = torch.ones([batch_size, self.n_steps], device=X.device)
        input_X = input_X.permute(0, 2, 1)

        # MOMENT backbone processing
        reconstruction = self.backbone(input_X, input_mask=input_mask).reconstruction
        # print(f"reconstruction.shape {reconstruction.shape}")
        reconstruction = reconstruction.permute(0, 2, 1)
        # print(f"reconstruction.shape {reconstruction.shape}")
        reconstruction = reconstruction[:, :, : self.n_features]

        # replace the observed part with values from X
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        # ensemble the results as a dictionary for return
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
