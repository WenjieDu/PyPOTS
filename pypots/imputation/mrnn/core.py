"""
The core wrapper assembles the submodules of M-RNN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.mrnn import BackboneMRNN


class _MRNN(ModelCore):
    def __init__(
        self,
        n_steps,
        n_features,
        rnn_hidden_size,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.backbone = BackboneMRNN(n_steps, n_features, rnn_hidden_size)
        self.training_loss = training_loss
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
        X = inputs["forward"]["X"]
        M = inputs["forward"]["missing_mask"]

        RNN_estimation, RNN_imputed_data, FCN_estimation = self.backbone(inputs)

        imputed_data = M * X + (1 - M) * FCN_estimation
        results = {
            "imputation": imputed_data,
            "RNN_estimation": RNN_estimation,
            "RNN_imputed_data": RNN_imputed_data,
            "FCN_estimation": FCN_estimation,
            "reconstruction": FCN_estimation,
        }

        if calc_criterion:
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                RNN_loss = self.training_loss(RNN_estimation, X, M)
                FCN_loss = self.training_loss(FCN_estimation, RNN_imputed_data)
                reconstruction_loss = RNN_loss + FCN_loss
                results["loss"] = reconstruction_loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
                results["metric"] = self.validation_metric(FCN_estimation, X_ori, indicating_mask)

        return results
