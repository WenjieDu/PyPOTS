"""
The core wrapper assembles the submodules of RevIN_SCINet imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.revin import RevIN
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding
from ...nn.modules.scinet import BackboneSCINet


class _RevIN_SCINet(ModelCore):
    def __init__(
        self,
        n_steps,
        n_features,
        n_stacks,
        n_levels,
        n_groups,
        n_decoder_layers,
        d_hidden,
        kernel_size,
        dropout,
        concat_len,
        pos_enc: bool,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            n_features,
            with_pos=False,
            dropout=dropout,
        )
        self.backbone = BackboneSCINet(
            n_out_steps=n_steps,
            n_in_steps=n_steps,
            n_in_features=n_features,
            d_hidden=d_hidden,
            n_stacks=n_stacks,
            n_levels=n_levels,
            n_decoder_layers=n_decoder_layers,
            n_groups=n_groups,
            kernel_size=kernel_size,
            dropout=dropout,
            concat_len=concat_len,
            modified=True,
            pos_enc=pos_enc,
            single_step_output_One=False,
        )
        self.revin = RevIN(n_features)

        # apply SAITS loss function to ReVIN SCINet on the imputation task
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
        X = self.revin(X, missing_mask, mode="norm")

        # WDU: the original RevIN_SCINet paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        enc_out = self.saits_embedding(X, missing_mask)

        # RevIN_SCINet encoder processing
        reconstruction, _ = self.backbone(enc_out)
        reconstruction = self.revin(reconstruction, mode="denorm")

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
