"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
from einops import rearrange, repeat

from ....nn.modules.transformer import ScaledDotProductAttention, MultiHeadAttention


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(
        self,
        seg_num,
        factor,
        d_model,
        n_heads,
        d_k,
        d_v,
        d_ff=None,
        dropout=0.1,
        attn_dropout=0.1,
    ):
        super().__init__()
        d_ff = 4 * d_model if d_ff is None else d_ff
        self.time_attention = MultiHeadAttention(
            ScaledDotProductAttention(d_k**0.5, attn_dropout),
            d_model,
            n_heads,
            d_k,
            d_v,
        )
        self.dim_sender = MultiHeadAttention(
            ScaledDotProductAttention(d_k**0.5, attn_dropout),
            d_model,
            n_heads,
            d_k,
            d_v,
        )
        self.dim_receiver = MultiHeadAttention(
            ScaledDotProductAttention(d_k**0.5, attn_dropout),
            d_model,
            n_heads,
            d_k,
            d_v,
        )
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch, ts_d, seg_num, d_model = x.shape
        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
        # time_in = x.reshape(-1, seg_num, d_model)
        time_enc, attn = self.time_attention(time_in, time_in, time_in, attn_mask=None)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross dimension stage: use a small set of learnable vectors to
        # aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model", b=batch)
        # dim_send = dim_in.reshape()
        batch_router = repeat(
            self.router,
            "seg_num factor d_model -> (repeat seg_num) factor d_model",
            repeat=batch,
        )
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, "(b seg_num) ts_d d_model -> b ts_d seg_num d_model", b=batch)

        return final_out


class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i :: self.win_size, :])
        x = torch.cat(seg_to_merge, -1)

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class ScaleBlock(nn.Module):
    def __init__(
        self,
        win_size,
        d_model,
        n_heads,
        d_ff,
        depth,
        dropout,
        seg_num,
        factor,
    ):
        super().__init__()

        d_k = d_model // n_heads
        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(
                TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_k, d_k, d_ff, dropout)
            )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x, None


class CrossformerDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, seg_len, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, "b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model")

        cross = rearrange(cross, "b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model")
        tmp, attn = self.cross_attention(
            x,
            cross,
            cross,
            None,
            None,
            None,
        )
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        dec_output = rearrange(
            dec_output,
            "(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model",
            b=batch,
        )
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, "b out_d seg_num seg_len -> b (out_d seg_num) seg_len")

        return dec_output, layer_predict
