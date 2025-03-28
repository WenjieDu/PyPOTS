"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inception import InceptionBlockV1, InceptionTransBlockV1


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    if len(frequency_list) < k:
        k = len(frequency_list)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    index = np.where(period > 0)
    top_list = top_list[index]
    period = period[period > 0]
    return period, abs(xf).mean(-1)[:, top_list], top_list


class RowAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super().__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        row_attn = torch.bmm(Q, K)
        row_attn = self.softmax(row_attn)
        out = torch.bmm(V, row_attn.permute(0, 2, 1))
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)
        out = self.gamma * out + x
        return out


class ColAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super().__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c2,h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c1,h)

        col_attn = torch.bmm(Q, K)
        col_attn = self.softmax(col_attn)
        out = torch.bmm(V, col_attn.permute(0, 2, 1))
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)
        out = self.gamma * out + x
        return out


class MultiScaleSeasonCross(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_kernels,
        downsampling_window,
    ):
        super().__init__()
        self.cross_conv_season = nn.Sequential(
            InceptionBlockV1(
                d_model,
                d_ff,
                num_kernels=num_kernels,
                stride=(downsampling_window, 1),
            ),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, season_list):
        B, N, _, _ = season_list[0].size()
        # cross high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 3, 1).reshape(B, -1, N)]
        for i in range(len(season_list) - 1):
            out_low_res = self.cross_conv_season(out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 3, 1).reshape(B, -1, N))
        return out_season_list


class MultiScaleTrendCross(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_kernels,
        downsampling_window,
    ):
        super().__init__()

        self.cross_trans_conv_season = InceptionTransBlockV1(
            d_model,
            d_ff,
            num_kernels=num_kernels,
            stride=(downsampling_window, 1),
        )
        self.cross_trans_conv_season_restore = nn.Sequential(
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, trend_list):
        B, N, _, _ = trend_list[0].size()
        # cross low->high
        trend_list.reverse()
        out_low = trend_list[0]
        out_high = trend_list[1]
        out_trend_list = [out_low.permute(0, 2, 3, 1).reshape(B, -1, N)]

        for i in range(len(trend_list) - 1):
            out_high_res = self.cross_trans_conv_season(out_low, output_size=out_high.size())
            out_high_res = self.cross_trans_conv_season_restore(out_high_res)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list) - 1:
                out_high = trend_list[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 3, 1).reshape(B, -1, N))

        out_trend_list.reverse()
        return out_trend_list


class MixerBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        top_k,
        d_model,
        d_ff,
        num_kernels,
        downsampling_window,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.downsampling_window = downsampling_window
        self.k = top_k
        self.layer_norm = nn.LayerNorm(d_model)
        self.row_attn_2d_trend = RowAttention(d_model, d_ff)
        self.col_attn_2d_season = ColAttention(d_model, d_ff)
        self.multi_scale_season_conv = MultiScaleSeasonCross(d_model, d_ff, num_kernels, downsampling_window)
        self.multi_scale_trend_conv = MultiScaleTrendCross(d_model, d_ff, num_kernels, downsampling_window)

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)
        period_list, period_weight, top_list = FFT_for_Period(x_list[-1], self.k)

        res_list = []
        for i in range(len(period_list)):
            period = period_list[i]
            season_list = []
            trend_list = []
            for x in x_list:
                out = self.time_imaging(x, period)
                season, trend = self.dual_axis_attn(out)
                season_list.append(season)
                trend_list.append(trend)

            out_list = self.multi_scale_mixing(season_list, trend_list, length_list)
            res_list.append(out_list)

        res_list_new = []
        for i in range(len(x_list)):
            list = []
            for j in range(len(period_list)):
                list.append(res_list[j][i])
            res = torch.stack(list, dim=-1)
            res_list_new.append(res)

        res_list_agg = []
        for x, res in zip(x_list, res_list_new):
            res = self.multi_reso_mixing(period_weight, x, res)
            res = self.layer_norm(res)
            res_list_agg.append(res)
        return res_list_agg

    def time_imaging(self, x, period):
        B, T, N = x.size()
        out, length = self.__conv_padding(x, period)
        out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
        return out

    def dual_axis_attn(self, out):
        trend = self.row_attn_2d_trend(out)
        season = self.col_attn_2d_season(out)
        return season, trend

    def multi_scale_mixing(self, season_list, trend_list, length_list):
        out_season_list = self.multi_scale_season_conv(season_list)
        out_trend_list = self.multi_scale_trend_conv(trend_list)
        out_list = []
        for out_season, out_trend, length in zip(out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out_list.append(out[:, :length, :])
        return out_list

    @staticmethod
    def __conv_padding(x, period, downsampling_window=1):
        B, T, N = x.size()

        if T % (period * downsampling_window) != 0:
            length = ((T // (period * downsampling_window)) + 1) * period * downsampling_window
            padding = torch.zeros([B, (length - T), N]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = T
            out = x
        return out, length

    @staticmethod
    def multi_reso_mixing(period_weight, x, res):
        B, T, N = x.size()
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
