"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .loss import BaseCriterion

from ..functional import calc_pr_auc, calc_acc


class PR_AUC(BaseCriterion):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        pr_auc, _, _, _ = calc_pr_auc(prediction, target)
        return pr_auc


class Accuracy(BaseCriterion):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        acc_score = calc_acc(prediction, target)
        return acc_score
