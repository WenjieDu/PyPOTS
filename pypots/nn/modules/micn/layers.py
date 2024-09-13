"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.fft
import torch.nn as nn

from ..autoformer import SeriesDecompositionBlock


class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(
        self,
        feature_size=512,
        decomp_kernel=[32],
        conv_kernel=[24],
        isometric_kernel=[18, 6],
    ):
        super().__init__()
        self.conv_kernel = conv_kernel

        # isometric convolution
        self.isometric_conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=i,
                    padding=0,
                    stride=1,
                )
                for i in isometric_kernel
            ]
        )

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=i,
                    padding=i // 2,
                    stride=i,
                )
                for i in conv_kernel
            ]
        )

        # upsampling convolution
        self.conv_trans = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=i,
                    padding=0,
                    stride=i,
                )
                for i in conv_kernel
            ]
        )

        self.decomp = nn.ModuleList([SeriesDecompositionBlock(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=(len(self.conv_kernel), 1),
        )

        # feedforward network
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=input.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]  # truncate

        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)

            # merge
        mg = torch.tensor([], device=src.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)

        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):
    def __init__(
        self,
        embedding_size=512,
        d_layers=1,
        decomp_kernel=[32],
        c_out=1,
        conv_kernel=[2, 4],
        isometric_kernel=[18, 6],
    ):
        super().__init__()

        self.mic = nn.ModuleList(
            [
                MIC(
                    feature_size=embedding_size,
                    decomp_kernel=decomp_kernel,
                    conv_kernel=conv_kernel,
                    isometric_kernel=isometric_kernel,
                )
                for _ in range(d_layers)
            ]
        )

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)
