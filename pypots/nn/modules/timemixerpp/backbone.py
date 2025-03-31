"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MixerBlock
from ..revin import RevIN
from ..transformer.attention import ScaledDotProductAttention, MultiHeadAttention
from ..transformer.embedding import DataEmbedding


class BackboneTimeMixerPP(nn.Module):
    def __init__(
        self,
        task_name: str,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_pred_features: int,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        dropout: float,
        top_k: int,
        n_kernels: int,
        channel_mixing: bool,
        channel_independence: bool,
        downsampling_layers: int,
        downsampling_window: int,
        downsampling_method: str,
        use_future_temporal_feature: bool,
        use_norm: bool = False,
        embed="fixed",
        freq="h",
        n_classes=None,
    ):
        super().__init__()
        self.task_name = task_name
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.n_layers = n_layers
        self.channel_mixing = channel_mixing
        self.channel_independence = channel_independence
        self.downsampling_window = downsampling_window
        self.downsampling_layers = downsampling_layers
        self.downsampling_method = downsampling_method
        self.use_norm = use_norm
        self.use_future_temporal_feature = use_future_temporal_feature

        assert downsampling_method in ["max", "avg", "conv"], "downsampling_method must be in ['max', 'avg', 'conv']"

        if self.channel_independence:
            self.enc_embedding = DataEmbedding(1, d_model, embed, freq, dropout, with_pos=False)
        else:
            self.enc_embedding = DataEmbedding(n_features, d_model, embed, freq, dropout, with_pos=False)

        if self.use_norm:
            self.revin_layers = torch.nn.ModuleList([RevIN(n_features) for _ in range(downsampling_layers + 1)])

        self.encoder_model = nn.ModuleList(
            [
                MixerBlock(
                    n_steps,
                    n_pred_steps,
                    top_k,
                    d_model,
                    d_ffn,
                    n_kernels,
                    downsampling_window,
                )
                for _ in range(n_layers)
            ]
        )

        if self.channel_mixing:
            assert n_steps >= downsampling_window**downsampling_layers
            d_time_model = n_steps // (downsampling_window**downsampling_layers)
            d_kv = d_time_model // n_heads
            full_attn_operator = ScaledDotProductAttention(d_kv**0.5, dropout)
            self.channel_mixing_attention = MultiHeadAttention(
                full_attn_operator,
                d_time_model,
                n_heads,
                d_kv,
                d_kv,
            )

        if self.downsampling_method == "max":
            self.down_pool = torch.nn.MaxPool1d(self.downsampling_window, return_indices=False)
        elif self.downsampling_method == "avg":
            self.down_pool = torch.nn.AvgPool1d(self.downsampling_window)
        elif self.downsampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2

            if self.channel_independence:
                in_channels = 1
            else:
                in_channels = n_features

            self.down_pool = nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=padding,
                stride=self.downsampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            raise ValueError("Downsampling method is error,only supporting the max, avg, conv1D")

        if task_name == "long_term_forecast" or task_name == "short_term_forecast":
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        n_steps // (downsampling_window**i),
                        n_pred_steps,
                    )
                    for i in range(downsampling_layers + 1)
                ]
            )

            if self.channel_independence:
                self.projection_layer = nn.Linear(d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(d_model, n_pred_features, bias=True)
        elif task_name == "imputation" or task_name == "anomaly_detection":
            if self.channel_independence:
                self.projection_layer = nn.Linear(d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(d_model, n_pred_features, bias=True)
        elif task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * n_steps, n_classes)
        else:
            raise NotImplementedError("Task not supported")

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # B,T,C -> B,C,T
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.downsampling_layers):
            if self.downsampling_method == "conv" and i == 0 and self.channel_independence:
                x_enc_ori = x_enc_ori.contiguous().reshape(B * N, T, 1).permute(0, 2, 1).contiguous()

            x_enc_sampling = self.down_pool(x_enc_ori)

            if self.downsampling_method == "conv":
                x_enc_sampling_list.append(
                    x_enc_sampling.reshape(B, N, T // (self.downsampling_window ** (i + 1))).permute(0, 2, 1)
                )
            else:
                x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))

            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, :: self.downsampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, :: self.downsampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()

                x = self.revin_layers[i](x, x_mark, mode="norm") if self.use_norm else x
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                x = self.revin_layers[i](x, mode="norm") if self.use_norm else x
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        if self.channel_mixing and self.channel_independence == 1:
            _, T, D = x_list[-1].size()

            coarse_scale_enc_out = x_list[-1].reshape(B, N, T * D)
            coarse_scale_enc_out, _ = self.channel_mixing_attention(
                coarse_scale_enc_out, coarse_scale_enc_out, coarse_scale_enc_out, None
            )
            x_list[-1] = coarse_scale_enc_out.reshape(B * N, T, D) + x_list[-1]

        enc_out_list = []
        if x_mark_enc is not None:
            for x, x_mark in zip(x_list, x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for x in x_list:
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        for i in range(self.n_layers):
            enc_out_list = self.encoder_model[i](enc_out_list)

        dec_out_list = []
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(B, self.n_pred_features, -1).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)

        dec_out = self.revin_layers[0](dec_out, mode="denorm") if self.use_norm else dec_out
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc
        enc_out_list = []

        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        for i in range(self.n_layer):
            enc_out_list = self.encoder_model[i](enc_out_list)

        enc_out = enc_out_list[0]
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc):
            B, T, N = x.size()
            x = self.revin_layers[i](x, "norm") if self.use_norm else x
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        if self.channel_mixing and self.channel_independence:
            _, T, D = x_list[-1].size()
            coarse_scale_enc_out = x_list[-1].reshape(B, N, T * D)
            coarse_scale_enc_out, _ = self.channel_mixing_attention(
                coarse_scale_enc_out, coarse_scale_enc_out, coarse_scale_enc_out, None
            )
            x_list[-1] = coarse_scale_enc_out.reshape(B * N, T, D) + x_list[-1]

        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        for i in range(self.n_layers):
            enc_out_list = self.encoder_model[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.revin_layers[0](dec_out, "denorm") if self.use_norm else dec_out
        return dec_out

    def imputation(self, x_enc, x_mark_enc):

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        if self.channel_mixing and self.channel_independence:
            _, T, D = x_list[-1].size()
            coarse_scale_enc_out = x_list[-1].reshape(B, N, T * D)
            coarse_scale_enc_out, _ = self.channel_mixing_attention(
                coarse_scale_enc_out, coarse_scale_enc_out, coarse_scale_enc_out, None
            )
            x_list[-1] = coarse_scale_enc_out.reshape(B * N, T, D) + x_list[-1]

        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        for i in range(self.n_layers):
            enc_out_list = self.encoder_model[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.n_pred_features, -1).permute(0, 2, 1).contiguous()

        return dec_out
