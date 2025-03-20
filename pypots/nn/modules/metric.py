"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .loss import Criterion

from ..functional import calc_pr_auc, calc_acc, calc_roc_auc


class PR_AUC(Criterion):
    def __init__(self, pos_label: int = 1):
        super().__init__(lower_better=False)
        self.pos_label = pos_label

    def forward(self, prediction, target):
        pr_auc, _, _, _ = calc_pr_auc(prediction, target, self.pos_label)
        return pr_auc


class ROC_AUC(Criterion):
    def __init__(self, pos_label: int = 1):
        super().__init__(lower_better=False)
        self.pos_label = pos_label

    def forward(self, prediction, target):
        roc_auc, _, _, _ = calc_roc_auc(prediction, target, self.pos_label)
        return roc_auc


class Accuracy(Criterion):
    def __init__(self):
        super().__init__(lower_better=False)

    def forward(self, prediction, target):
        acc_score = calc_acc(prediction, target)
        return acc_score
