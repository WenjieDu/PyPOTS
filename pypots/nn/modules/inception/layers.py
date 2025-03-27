"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn


class InceptionBlockV1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_kernels=6,
        stride=1,
        init_weight=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.stride = stride
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,
                    padding=i,
                    stride=stride,
                )
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class InceptionTransBlockV1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        num_kernels=6,
        init_weight=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.stride = stride

        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,
                    padding=i,
                    stride=stride,
                )
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, output_size):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x, output_size=output_size))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
