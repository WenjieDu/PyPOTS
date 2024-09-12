"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super().__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi, self.multi * self.time_step)
        )  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
            else:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        # ffted = torch.fft.rfft(input, 1, onesided=False) # original old version, onesided doesn't work in new torch
        ffted = torch.view_as_real(torch.fft.fft(input, dim=1))  # WDU: replace the above line with this line
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        # iffted = torch.fft.irfft(time_step_as_inner, 1, onesided=False)  # onesided doesn't work in new torch
        iffted = torch.fft.irfft(
            torch.view_as_complex(time_step_as_inner),
            n=time_step_as_inner.shape[1],
            dim=1,
        )
        return iffted

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(mul_L, x)
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        igfted = torch.matmul(gconv_input, self.weight)
        igfted = torch.sum(igfted, dim=1)
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source
