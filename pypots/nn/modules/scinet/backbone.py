"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .layers import EncoderTree


class BackboneSCINet(nn.Module):
    def __init__(
        self,
        output_len,
        input_len,
        input_dim=9,
        hid_size=1,
        num_stacks=1,
        num_levels=3,
        num_decoder_layer=1,
        concat_len=0,
        groups=1,
        kernel=5,
        dropout=0.5,
        single_step_output_One=0,
        positionalE=False,
        modified=True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE
        self.num_decoder_layer = num_decoder_layer

        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels=self.num_levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            groups=self.groups,
            hidden_size=self.hidden_size,
            INN=modified,
        )

        if num_stacks == 2:  # we only implement two stacks at most.
            self.blocks2 = EncoderTree(
                in_planes=self.input_dim,
                num_levels=self.num_levels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                groups=self.groups,
                hidden_size=self.hidden_size,
                INN=modified,
            )

        self.stacks = num_stacks

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(
            self.input_len, self.output_len, kernel_size=1, stride=1, bias=False
        )
        self.div_projection = nn.ModuleList()
        self.overlap_len = self.input_len // 4
        self.div_len = self.input_len // 6

        if self.num_decoder_layer > 1:
            self.projection1 = nn.Linear(self.input_len, self.output_len)
            for layer_idx in range(self.num_decoder_layer - 1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = (
                        min(i * self.div_len + self.overlap_len, self.input_len)
                        - i * self.div_len
                    )
                    div_projection.append(nn.Linear(lens, self.div_len))
                self.div_projection.append(div_projection)

        if self.single_step_output_One:  # only output the N_th timestep.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(
                        self.concat_len + self.output_len, 1, kernel_size=1, bias=False
                    )
                else:
                    self.projection2 = nn.Conv1d(
                        self.input_len + self.output_len, 1, kernel_size=1, bias=False
                    )
        else:  # output the N timesteps.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(
                        self.concat_len + self.output_len,
                        self.output_len,
                        kernel_size=1,
                        bias=False,
                    )
                else:
                    self.projection2 = nn.Conv1d(
                        self.input_len + self.output_len,
                        self.output_len,
                        kernel_size=1,
                        bias=False,
                    )

        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1

        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / max(num_timescales - 1, 1)
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        self.register_buffer("inv_timescales", inv_timescales)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(
            max_length, dtype=torch.float32, device=x.device
        )  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        )  # [T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)

        return signal

    def forward(self, x):
        assert (
            self.input_len % (np.power(2, self.num_levels)) == 0
        )  # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        if self.num_decoder_layer == 1:
            x = self.projection1(x)
        else:
            x = x.permute(0, 2, 1)
            for div_projection in self.div_projection:
                output = torch.zeros(x.shape, dtype=x.dtype).cuda()
                for i, div_layer in enumerate(div_projection):
                    div_x = x[
                        :,
                        :,
                        i
                        * self.div_len : min(
                            i * self.div_len + self.overlap_len, self.input_len
                        ),
                    ]
                    output[:, :, i * self.div_len : (i + 1) * self.div_len] = div_layer(
                        div_x
                    )
                x = output
            x = self.projection1(x)
            x = x.permute(0, 2, 1)

        if self.stacks == 1:
            return x, None

        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len :, :], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)

            # the second stack
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)
            return x, MidOutPut
