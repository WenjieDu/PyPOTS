"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional

import torch
import torch.nn as nn

from ...autoformer.modules.submodules import SeriesDecompositionBlock
from ....utils.metrics import calc_mse


class _DLinear(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        moving_avg_window_size: int,
        individual: bool = False,
        d_model: Optional[int] = None,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.individual = individual
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.series_decomp = SeriesDecompositionBlock(moving_avg_window_size)

        if individual:
            # create linear layers for each feature individually
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for i in range(n_features):
                self.linear_seasonal.append(nn.Linear(n_steps, n_steps))
                self.linear_trend.append(nn.Linear(n_steps, n_steps))
                self.linear_seasonal[i].weight = nn.Parameter(
                    (1 / n_steps) * torch.ones([n_steps, n_steps])
                )
                self.linear_trend[i].weight = nn.Parameter(
                    (1 / n_steps) * torch.ones([n_steps, n_steps])
                )
        else:
            if d_model is None:
                raise ValueError(
                    "The argument d_model is necessary for DLinear in the non-individual mode."
                )
            self.linear_seasonal = nn.Linear(n_steps, n_steps)
            self.linear_trend = nn.Linear(n_steps, n_steps)
            self.linear_seasonal.weight = nn.Parameter(
                (1 / n_steps) * torch.ones([n_steps, n_steps])
            )
            self.linear_trend.weight = nn.Parameter(
                (1 / n_steps) * torch.ones([n_steps, n_steps])
            )

            self.linear_seasonal_embedding = nn.Linear(n_features * 2, d_model)
            self.linear_trend_embedding = nn.Linear(n_features * 2, d_model)
            self.linear_seasonal_output = nn.Linear(d_model, n_features)
            self.linear_trend_output = nn.Linear(d_model, n_features)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        # input preprocessing and embedding for DLinear
        seasonal_init, trend_init = self.series_decomp(X)

        # DLinear processing
        if self.individual:
            seasonal_init, trend_init = seasonal_init.permute(
                0, 2, 1
            ), trend_init.permute(0, 2, 1)
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.n_steps],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.n_steps],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.n_features):
                seasonal_output[:, i, :] = self.linear_seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])

            seasonal_output = seasonal_output.permute(0, 2, 1)
            trend_output = trend_output.permute(0, 2, 1)
        else:
            # WDU: the original DLinear paper isn't proposed for imputation task. Hence the model doesn't take
            # the missing mask into account, which means, in the process, the model doesn't know which part of
            # the input data is missing, and this may hurt the model's imputation performance. Therefore, I add the
            # embedding layers to project the concatenation of features and masks into a hidden space, as well as
            # the output layers to project the seasonal and trend from the hidden space to the original space.
            # But this is only for the non-individual mode.
            seasonal_init = torch.cat([seasonal_init, masks], dim=2)
            trend_init = torch.cat([trend_init, masks], dim=2)
            seasonal_init = self.linear_seasonal_embedding(seasonal_init)
            trend_init = self.linear_trend_embedding(trend_init)
            seasonal_init, trend_init = seasonal_init.permute(
                0, 2, 1
            ), trend_init.permute(0, 2, 1)

            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)
            seasonal_output = seasonal_output.permute(0, 2, 1)
            trend_output = trend_output.permute(0, 2, 1)
            seasonal_output = self.linear_seasonal_output(seasonal_output)
            trend_output = self.linear_trend_output(trend_output)

        output = seasonal_output + trend_output

        imputed_data = masks * X + (1 - masks) * output
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # apply SAITS loss function to DLinear on the imputation task
            ORT_loss = calc_mse(output, X, masks)
            MIT_loss = calc_mse(output, inputs["X_ori"], inputs["indicating_mask"])
            # `loss` is always the item for backward propagating to update the model
            loss = self.ORT_weight * ORT_loss + self.MIT_weight * MIT_loss
            results["loss"] = loss

        return results
