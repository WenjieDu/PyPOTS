"""
The implementation of the modules for BRITS.

Refer to the paper "Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018."

Notes
-----
Partial implementation uses code from https://github.com/caow13/BRITS. The bugs in the original implementation
are fixed here.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from typing import Union

import torch
import torch.nn as nn

from ...imputation.brits.model import (
    RITS as imputation_RITS,
)


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
