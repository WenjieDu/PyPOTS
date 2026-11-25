"""Implementation of pypots.nn.modules.saits.loss."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from torch import nn

from ..loss import MAE, Criterion


class SaitsLoss(nn.Module):
    def __init__(
        self,
        ORT_weight,
        MIT_weight,
        loss_calc_func: Criterion | None = None,
    ):
        super().__init__()
        if loss_calc_func is None:
            loss_calc_func = MAE()
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.loss_calc_func = loss_calc_func

    def forward(self, reconstruction, X_ori, missing_mask, indicating_mask):
        # calculate loss for the observed reconstruction task (ORT)
        ORT_loss = self.ORT_weight * self.loss_calc_func(reconstruction, X_ori, missing_mask)
        # calculate loss for the masked imputation task (MIT)
        MIT_loss = self.MIT_weight * self.loss_calc_func(reconstruction, X_ori, indicating_mask)
        # calculate the loss to back propagate for model updating
        loss = ORT_loss + MIT_loss
        return loss, ORT_loss, MIT_loss
