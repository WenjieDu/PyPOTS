"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch.nn as nn

from ..functional import calc_pr_auc, calc_acc


class BaseMetric(nn.Module):
    def __init__(self, lower_better: bool = True):
        super().__init__()
        self.lower_better = lower_better

    def forward(self, prediction, target):
        raise NotImplementedError


class PR_AUC(BaseMetric):
    def __init__(self):
        super().__init__(lower_better=False)

    def forward(self, prediction, target):
        pr_auc, _, _, _ = calc_pr_auc(prediction, target)
        return pr_auc


class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__(lower_better=False)

    def forward(self, prediction, target):
        acc_score = calc_acc(prediction, target)
        return acc_score
