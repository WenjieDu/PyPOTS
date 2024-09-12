"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional

import torch
import torch.nn as nn


class BackboneDLinear(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        individual: bool = False,
        d_model: Optional[int] = None,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.individual = individual

        if individual:
            # create linear layers for each feature individually
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for i in range(n_features):
                self.linear_seasonal.append(nn.Linear(n_steps, n_steps))
                self.linear_trend.append(nn.Linear(n_steps, n_steps))
                self.linear_seasonal[i].weight = nn.Parameter((1 / n_steps) * torch.ones([n_steps, n_steps]))
                self.linear_trend[i].weight = nn.Parameter((1 / n_steps) * torch.ones([n_steps, n_steps]))
        else:
            if d_model is None:
                raise ValueError("The argument d_model is necessary for DLinear in the non-individual mode.")
            self.linear_seasonal = nn.Linear(n_steps, n_steps)
            self.linear_trend = nn.Linear(n_steps, n_steps)
            self.linear_seasonal.weight = nn.Parameter((1 / n_steps) * torch.ones([n_steps, n_steps]))
            self.linear_trend.weight = nn.Parameter((1 / n_steps) * torch.ones([n_steps, n_steps]))

    def forward(self, seasonal_init, trend_init):
        if self.individual:
            seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.n_steps],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.n_steps],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.n_features):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])

            seasonal_output = seasonal_output.permute(0, 2, 1)
            trend_output = trend_output.permute(0, 2, 1)
        else:
            seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)
            seasonal_output = seasonal_output.permute(0, 2, 1)
            trend_output = trend_output.permute(0, 2, 1)

        return seasonal_output, trend_output
