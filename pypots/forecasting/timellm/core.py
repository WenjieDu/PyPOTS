"""
The core wrapper assembles the submodules of TimeLLM forecasting model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.timellm import BackboneTimeLLM


class _TimeLLM(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_pred_features: int,
        term: str,
        n_layers: int,
        patch_size: int,
        patch_stride: int,
        d_model: int,
        d_ffn: int,
        d_llm: int,
        n_heads: int,
        llm_model_type: str,
        dropout: float,
        domain_prompt_content: str,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        assert term in ["long", "short"], "forecasting term should be either 'long' or 'short'"
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.backbone = BackboneTimeLLM(
            n_steps,
            n_features,
            n_pred_steps,
            n_layers,
            patch_size,
            patch_stride,
            d_model,
            d_ffn,
            d_llm,
            n_heads,
            llm_model_type,
            dropout,
            domain_prompt_content,
            term + "_term_forecast",
        ).float()

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # TimeLLM processing
        forecasting_result = self.backbone(X, missing_mask)
        # the raw output has length = n_steps+n_pred_steps, we only need the last n_pred_steps
        forecasting_result = forecasting_result[:, -self.n_pred_steps :]

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
