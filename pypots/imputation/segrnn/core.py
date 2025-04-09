"""
The core wrapper assembles the submodules of SegRNN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Shengsheng Lin

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding
from ...nn.modules.segrnn import BackboneSegRNN


class _SegRNN(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        seg_len: int,
        d_model: int,
        dropout: float,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.seg_len = seg_len
        self.d_model = d_model
        self.dropout = dropout

        self.embedding = SaitsEmbedding(
            n_features * 2,
            n_features,
            with_pos=False,
        )
        self.backbone = BackboneSegRNN(
            n_steps,
            n_features,
            n_steps,
            seg_len,
            d_model,
            dropout,
        )

        # apply SAITS loss function to SegRNN on the imputation task
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

        X = self.embedding(X, missing_mask)
        reconstruction = self.backbone(X)

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
