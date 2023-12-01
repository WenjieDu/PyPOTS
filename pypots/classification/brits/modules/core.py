"""
The implementation of BRITS for the partially-observed time-series classification task.

Refer to the paper "Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018."

Notes
-----
Partial implementation uses code from https://github.com/caow13/BRITS. The bugs in the original implementation
are fixed here.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....imputation.brits.modules.core import RITS as imputation_RITS
from ....imputation.brits.modules.core import _BRITS as imputation_BRITS


class RITS(imputation_RITS):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        device: Union[str, torch.device],
    ):
        super().__init__(n_steps, n_features, rnn_hidden_size, device)
        self.dropout = nn.Dropout(p=0.25)
        self.classifier = nn.Linear(self.rnn_hidden_size, n_classes)

    def forward(self, inputs: dict, direction: str = "forward") -> dict:
        ret_dict = super().forward(inputs, direction)
        logits = self.classifier(ret_dict["final_hidden_state"])
        ret_dict["prediction"] = torch.softmax(logits, dim=1)
        return ret_dict


class _BRITS(imputation_BRITS, nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        classification_weight: float,
        reconstruction_weight: float,
        device: Union[str, torch.device],
    ):
        super().__init__(n_steps, n_features, rnn_hidden_size, device)
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes

        # create models
        self.rits_f = RITS(n_steps, n_features, rnn_hidden_size, n_classes, device)
        self.rits_b = RITS(n_steps, n_features, rnn_hidden_size, n_classes, device)
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight

    def impute(self, inputs: dict) -> torch.Tensor:
        return super().impute(inputs)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        """Forward processing of BRITS.

        Parameters
        ----------
        inputs :
            The input data.

        training :
            Whether in training mode.

        Returns
        -------
        dict, A dictionary includes all results.
        """
        ret_f = self.rits_f(inputs, "forward")
        ret_b = self._reverse(self.rits_b(inputs, "backward"))

        classification_pred = (ret_f["prediction"] + ret_b["prediction"]) / 2
        results = {"classification_pred": classification_pred}

        # if in training mode, return results with losses
        if training:
            ret_f["classification_loss"] = F.nll_loss(
                torch.log(ret_f["prediction"]), inputs["label"]
            )
            ret_b["classification_loss"] = F.nll_loss(
                torch.log(ret_b["prediction"]), inputs["label"]
            )
            consistency_loss = self._get_consistency_loss(
                ret_f["imputed_data"], ret_b["imputed_data"]
            )
            classification_loss = (
                ret_f["classification_loss"] + ret_b["classification_loss"]
            ) / 2
            reconstruction_loss = (
                ret_f["reconstruction_loss"] + ret_b["reconstruction_loss"]
            ) / 2

            results["consistency_loss"] = consistency_loss
            results["classification_loss"] = classification_loss
            results["reconstruction_loss"] = reconstruction_loss

            # `loss` is always the item for backward propagating to update the model
            loss = (
                consistency_loss
                + reconstruction_loss * self.reconstruction_weight
                + classification_loss * self.classification_weight
            )
            results["loss"] = loss

        return results
