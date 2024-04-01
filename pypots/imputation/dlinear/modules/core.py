"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

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
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.series_decomp = SeriesDecompositionBlock(moving_avg_window_size)
        self.individual = individual

        if individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.n_features):
                self.Linear_Seasonal.append(nn.Linear(self.n_steps, self.n_steps))
                self.Linear_Trend.append(nn.Linear(self.n_steps, self.n_steps))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.n_steps) * torch.ones([self.n_steps, self.n_steps])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.n_steps) * torch.ones([self.n_steps, self.n_steps])
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.n_steps, self.n_steps)
            self.Linear_Trend = nn.Linear(self.n_steps, self.n_steps)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.n_steps) * torch.ones([self.n_steps, self.n_steps])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.n_steps) * torch.ones([self.n_steps, self.n_steps])
            )

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        # DLinear encoder processing
        seasonal_init, trend_init = self.series_decomp(X)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.n_steps],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.n_steps],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.n_features):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        output = seasonal_output + trend_output
        output = output.permute(0, 2, 1)

        imputed_data = masks * X + (1 - masks) * output
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(output, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
