"""
The layers and building blocks of UniTS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    """Standard MLP block used in transformer architectures."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicLinear(nn.Module):
    """A dynamic linear layer that can interpolate the weight size to support any given input and output feature
    dimension."""

    def __init__(self, in_features=None, out_features=None, fixed_in=0, bias=True):
        super().__init__()
        assert fixed_in < in_features, "fixed_in < in_features is required"
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.fixed_in = fixed_in
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, out_features):
        fixed_weights = self.weights[:, : self.fixed_in]
        dynamic_weights = self.weights[:, self.fixed_in :]
        this_bias = self.bias
        in_features = x.shape[-1]

        if in_features != self.weights.size(1) or out_features != self.weights.size(0):
            dynamic_weights = F.interpolate(
                dynamic_weights.unsqueeze(0).unsqueeze(0),
                size=(out_features, in_features - self.fixed_in),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            if self.fixed_in != 0:
                fixed_weights = F.interpolate(
                    fixed_weights.unsqueeze(0).unsqueeze(0),
                    size=(out_features, self.fixed_in),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
        if out_features != self.weights.size(0):
            this_bias = F.interpolate(
                this_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                size=(1, out_features),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0).squeeze(0)
        return F.linear(x, torch.cat((fixed_weights, dynamic_weights), dim=1), this_bias)


class DynamicLinearMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        prefix_token_length=None,
        group=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # handle bias as tuple
        if isinstance(bias, (list, tuple)):
            bias_0, bias_1 = bias[0], bias[1]
        else:
            bias_0 = bias_1 = bias
        # handle drop as tuple
        if isinstance(drop, (list, tuple)):
            drop_0, drop_1 = drop[0], drop[1]
        else:
            drop_0 = drop_1 = drop

        self.fc1 = nn.Conv1d(in_features, hidden_features, 3, groups=group, bias=bias_0, padding=1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_0)

        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.seq_fc = DynamicLinear(
            hidden_features // 4, hidden_features // 4, bias=bias_1, fixed_in=prefix_token_length
        )
        self.prompt_fc = DynamicLinear(
            hidden_features // 4, prefix_token_length, bias=bias_1, fixed_in=prefix_token_length
        )

        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias_1)
        self.drop2 = nn.Dropout(drop_1)
        self.hidden_features = hidden_features
        self.prefix_token_length = prefix_token_length

    def dynamic_linear(self, x, prefix_seq_len):
        x_func = x[:, :, prefix_seq_len:]
        x_seq = x[:, :, :prefix_seq_len]
        x_seq_out = self.seq_fc(x_seq, x_seq.shape[-1] - self.prefix_token_length)
        x_prompt = self.prompt_fc(x_seq, self.prefix_token_length)
        x = torch.cat((x_prompt, x_seq_out, x_func), dim=-1)
        return x

    def split_dynamic_linear(self, x, prefix_seq_len):
        x1, x2 = x.chunk(2, dim=-2)
        x1 = self.dynamic_linear(x1, prefix_seq_len)
        return torch.cat((x1, x2), dim=-2)

    def forward(self, x, prefix_seq_len=None):
        n, var, l, c = x.shape
        x = x.view(-1, l, c)
        x = x.transpose(-1, -2)
        x = self.fc1(x)
        x = self.split_dynamic_linear(x, prefix_seq_len)
        x = self.act(x)
        x = self.drop1(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.fc2(x).view(n, var, l, c)
        x = self.drop2(x)
        return x


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, 1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        return self.pe[:, :, offset : offset + x.size(2)]


class GateLayer(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        return self.gate(x).sigmoid() * x


class SeqAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, P, C = x.shape

        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads, self.head_dim).permute(3, 0, 2, 4, 1, 5)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.mean(dim=1, keepdim=False)
        k = k.mean(dim=1, keepdim=False)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.view(B, self.num_heads, N, -1, P).permute(0, 2, 4, 1, 3).reshape(B, N, P, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqAttBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask):
        x_input = x
        x = self.norm1(x)
        n_vars, n_seqs = x.shape[1], x.shape[2]
        x = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))
        x = self.attn_seq(x, attn_mask)
        x = torch.reshape(x, (-1, n_vars, n_seqs, x.shape[-1]))
        x = x_input + self.drop_path1(self.ls1(x))
        return x


class VarAttBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn_var(self.norm1(x))))
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        proj_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=None,
        prefix_token_length=0,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        if mlp_layer is DynamicLinearMlp:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
                prefix_token_length=prefix_token_length,
            )
        else:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )
        self.ls2 = GateLayer(dim, init_values=init_values)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, prefix_seq_len=None):
        if prefix_seq_len is not None:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), prefix_seq_len=prefix_seq_len)))
        else:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=8.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        prefix_token_length=0,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            init_values=init_values,
            proj_drop=proj_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
        )
        self.var_att_block = VarAttBlock(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            init_values=init_values,
            proj_drop=proj_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
        )
        self.dynamic_mlp = MLPBlock(
            dim=dim,
            mlp_ratio=mlp_ratio,
            mlp_layer=DynamicLinearMlp,
            proj_drop=proj_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            prefix_token_length=prefix_token_length,
        )

    def forward(self, x, prefix_seq_len, attn_mask):
        x = self.seq_att_block(x, attn_mask)
        x = self.var_att_block(x)
        x = self.dynamic_mlp(x, prefix_seq_len=prefix_seq_len)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        assert self.patch_len == self.stride, "non-overlap patching required (patch_len == stride)"
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class ForecastHead(nn.Module):
    def __init__(self, d_model, patch_len, stride, pad, head_dropout=0, prefix_token_length=None):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=int(d_model * 4),
            act_layer=nn.GELU,
            drop=head_dropout,
        )
        self.proj_out = nn.Linear(d_model, patch_len)
        self.pad = pad
        self.patch_len = patch_len
        self.stride = stride
        self.pos_proj = DynamicLinear(in_features=128, out_features=128, fixed_in=prefix_token_length)

    def forward(self, x_full, pred_len, token_len):
        x_full = self.proj_in(x_full)
        x_pred = x_full[:, :, -token_len:]
        x = x_full.transpose(-1, -2)
        x = self.pos_proj(x, token_len)
        x = x.transpose(-1, -2)
        x = x + x_pred
        x = self.mlp(x)
        x = self.proj_out(x)

        bs, n_vars = x.shape[0], x.shape[1]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.fold(
            x,
            output_size=(pred_len, 1),
            kernel_size=(self.patch_len, 1),
            stride=(self.stride, 1),
        )
        x = x.squeeze(dim=-1)
        x = x.reshape(bs, n_vars, -1)
        x = x.permute(0, 2, 1)
        return x
