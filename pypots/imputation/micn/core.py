"""
The core wrapper assembles the submodules of MICN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.modules import ModelCore
from ...nn.modules.fedformer.layers import SeriesDecompositionMultiBlock
from ...nn.modules.loss import Criterion
from ...nn.modules.micn import BackboneMICN
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding


class _MICN(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        dropout: float,
        conv_kernel: list,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

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
            n_steps,
            n_features,
            n_layers,
            d_model,
            decomp_kernel,
            isometric_kernel,
            conv_kernel,
        )

        # apply SAITS loss function to MICN on the imputation task
        self.training_loss = SaitsLoss(ORT_weight, MIT_weight, training_loss)
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        seasonal_init, trend_init = self.decomp_multi(X)

        # WDU: the original MICN paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        enc_out = self.saits_embedding(seasonal_init, missing_mask)

        # MICN encoder processing
        reconstruction = self.backbone(enc_out)
        reconstruction = reconstruction + trend_init

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
            "reconstruction": reconstruction,
        }

        return results

    def calc_criterion(self, inputs: dict) -> dict:
        results = self.forward(inputs)

        X_ori, indicating_mask, missing_mask = inputs["X_ori"], inputs["indicating_mask"], inputs["missing_mask"]
        reconstruction = results["reconstruction"]

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
