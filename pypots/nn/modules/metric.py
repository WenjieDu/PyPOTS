"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import torch

from .loss import Criterion
from ..functional import (
    calc_acc,
    calc_pr_auc,
    calc_roc_auc,
)


class PR_AUC(Criterion):
    def __init__(self, pos_label: int = 1):
        super().__init__(lower_better=False)
        self.pos_label = pos_label

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        assert len(logits.shape) == 2 and logits.shape[1] > 1
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()

        binary_prediction_proba = probabilities[:, self.pos_label]

        pr_auc, _, _, _ = calc_pr_auc(binary_prediction_proba, targets, self.pos_label)
        return torch.FloatTensor([pr_auc])


class ROC_AUC(Criterion):
    def __init__(self, pos_label: int = 1):
        super().__init__(lower_better=False)
        self.pos_label = pos_label

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        roc_auc, _, _, _ = calc_roc_auc(probabilities, targets, self.pos_label)
        return torch.FloatTensor([roc_auc])


class Accuracy(Criterion):
    def __init__(self):
        super().__init__(lower_better=False)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        class_predictions = np.argmax(probabilities, axis=1)
        targets = targets.cpu().numpy()
        acc_score = calc_acc(class_predictions, targets)
        return torch.FloatTensor([acc_score])
