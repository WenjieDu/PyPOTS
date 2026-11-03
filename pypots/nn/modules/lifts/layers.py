"""Components adapted from the official LIFTS implementation.

Source: https://github.com/sanyouwu/lifts-open-source
Reference revision: dd03fd1ae1cf426251dc434c4dd3ba323f2e1d40
Copyright (c) 2026 LIFTS contributors. MIT license; see LICENSE in this directory.
Attention components also retain the SAITS MIT notice in LICENSE.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class RoPE1D(nn.Module):
    """RoPE1D component from the reference implementation."""

    def __init__(self, dim, max_len=48, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_len = max_len
        inv_freq = 1.0 / self.base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_len)

    def _set_cos_sin_cache(self, seq_len):
        self.max_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, start_pos=0):
        _batch, _n_heads, seq_len, head_dim = x.shape
        end_pos = start_pos + seq_len
        if end_pos > self.max_len:
            self._set_cos_sin_cache(end_pos + 50)
        cos = self.cos_cached[start_pos:end_pos].view(1, 1, seq_len, head_dim)
        sin = self.sin_cached[start_pos:end_pos].view(1, 1, seq_len, head_dim)
        return self._apply_rotary_pos_emb(x, cos, sin)

    def _apply_rotary_pos_emb(self, x, cos, sin):

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        return x * cos + rotate_half(x) * sin


def get_alibi_slopes(num_heads):
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** (-(math.log2(closest_power_of_2) - 3))))
    powers = torch.arange(1, closest_power_of_2 + 1)
    slopes = base**powers
    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** (-(math.log2(2 * closest_power_of_2) - 3))))
        extra_powers = torch.arange(1, 2 * (num_heads - closest_power_of_2) + 1, 2)
        extra_slopes = extra_base**extra_powers
        slopes = torch.cat([slopes, extra_slopes])
    return slopes


def get_alibi_bias(seq_len, num_heads, device):
    slopes = get_alibi_slopes(num_heads).to(device)
    positions = torch.arange(seq_len, device=device)
    pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    alibi_bias = slopes.unsqueeze(1).unsqueeze(2) * pos_diff.unsqueeze(0)
    return alibi_bias


def get_alibi_bias_1d(seq_len, num_heads, device, query_pos=None):
    slopes = get_alibi_slopes(num_heads).to(device)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    if query_pos is not None:
        pos_diff = positions - query_pos
    else:
        pos_diff = positions
    alibi_bias = slopes.unsqueeze(1) * pos_diff.unsqueeze(0)
    return alibi_bias.unsqueeze(1)


class ScaledDotProductAttention(nn.Module):
    """ScaledDotProductAttention component from the reference implementation."""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None, alibi_bias=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if alibi_bias is not None:
            attn = attn + alibi_bias
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return (output, attn)


class RoPEMultiHeadAttention(nn.Module):
    """RoPEMultiHeadAttention component from the reference implementation."""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout, max_len=48, rope_base=2000):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.rope = RoPE1D(dim=d_k, max_len=max_len, base=rope_base)
        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None, alibi_bias=None, start_pos_q=0, start_pos_k=0):
        d_k, d_v, n_head = (self.d_k, self.d_v, self.n_head)
        sz_b, len_q, len_k, len_v = (q.size(0), q.size(1), k.size(1), v.size(1))
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        q = self.rope(q, start_pos=start_pos_q)
        k = self.rope(k, start_pos=start_pos_k)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
        if alibi_bias is not None and alibi_bias.dim() == 3:
            alibi_bias = alibi_bias.unsqueeze(0)
        v, attn_weights = self.attention(q, k, v, attn_mask, alibi_bias=alibi_bias)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return (v, attn_weights)


class PositionWiseFeedForward(nn.Module):
    """PositionWiseFeedForward component from the reference implementation."""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class RoPEEncoderLayer(nn.Module):
    """RoPEEncoderLayer component from the reference implementation."""

    def __init__(
        self, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, attn_dropout=0.1, max_len=50, **kwargs
    ):
        super().__init__()
        self.diagonal_attention_mask = kwargs.get("diagonal_attention_mask", True)
        self.d_time = d_time
        self.d_feature = d_feature
        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = RoPEMultiHeadAttention(
            n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, attn_dropout=attn_dropout, max_len=max_len
        )
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, start_pos=0):
        current_T = enc_input.size(1)
        if self.diagonal_attention_mask:
            mask_time = ~torch.tril(torch.ones(current_T, current_T)).bool().to(enc_input.device)
        else:
            mask_time = None
        residual = enc_input
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(
            q=enc_input,
            k=enc_input,
            v=enc_input,
            attn_mask=mask_time,
            alibi_bias=None,
            start_pos_q=start_pos,
            start_pos_k=start_pos,
        )
        enc_output = self.dropout(enc_output)
        enc_output += residual
        enc_output = self.pos_ffn(enc_output)
        return (enc_output, attn_weights)


class InitialNoiseProcessorLearnable(nn.Module):
    """InitialNoiseProcessorLearnable component from the reference implementation."""

    def __init__(self, d_model, hidden_ratio=5):
        super().__init__()
        self.d_model = d_model
        hidden_dim = int(d_model * hidden_ratio)
        self.noise_mlp = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model))
        nn.init.normal_(self.noise_mlp[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.noise_mlp[-1].bias)

    def forward(self, batch_size, device):
        z_noise = torch.randn(batch_size, self.d_model, device=device)
        perturbation = self.noise_mlp(z_noise)
        h_0 = perturbation
        h_0 = torch.clamp(h_0, min=-10.0, max=10.0)
        return h_0.unsqueeze(1)


class SSM_Filter(nn.Module):
    """SSM Filter component from the reference implementation."""

    def __init__(
        self,
        time_dim,
        feature_dim,
        keep_dim,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        n_groups,
        n_group_inner_layers,
        **kwargs,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.keep_dim = keep_dim
        self.d_model = d_model
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        if "diagonal_attention_mask" in kwargs:
            kwargs.pop("diagonal_attention_mask")
        self.encoder_layer_stack = nn.ModuleList(
            [
                RoPEEncoderLayer(
                    d_time=time_dim,
                    d_feature=2 * feature_dim,
                    d_model=d_model,
                    d_inner=d_inner,
                    n_head=n_head,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    attn_dropout=dropout,
                    max_len=time_dim + 2,
                    diagonal_attention_mask=True,
                    **kwargs,
                )
                for _ in range(n_groups)
            ]
        )
        self.encoder_embedding = nn.Linear(2 * feature_dim, d_model)
        self.mask_proj = nn.Sequential(nn.Linear(feature_dim, d_model), nn.GELU())
        self.head_trend = nn.Linear(d_model, keep_dim)
        self.noise_fusion = nn.Sequential(
            nn.Linear(2 * d_model + keep_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU()
        )
        self.head_residual = nn.Linear(d_model, keep_dim)
        nn.init.normal_(self.head_residual.weight, mean=0.0, std=0.05)
        nn.init.constant_(self.head_residual.bias, 0.0)

    def forward(self, x, mask):
        B, T, _ = x.shape
        device = x.device
        x_enc = x
        mask_enc = mask.float()
        x_noise = x_enc * mask_enc + (1 - mask_enc) * torch.randn_like(x_enc)
        if mask_enc.all():
            noise_channel = torch.randn_like(x_enc)
            masked_x = torch.cat([x_noise, noise_channel], dim=-1)
        else:
            masked_x = torch.cat([x_noise, mask_enc], dim=-1)
        Z_feat = self.encoder_embedding(masked_x)
        for encoder_layer in self.encoder_layer_stack:
            for _ in range(self.n_group_inner_layers):
                Z_feat, _ = encoder_layer(Z_feat)
        z_trend = self.head_trend(Z_feat)
        mask_feat = self.mask_proj(mask_enc)
        noise = torch.randn(B, T, self.keep_dim, device=device)
        fusion_input = torch.cat([Z_feat, noise, mask_feat], dim=-1)
        noise_feat = self.noise_fusion(fusion_input)
        z_residual = self.head_residual(noise_feat)
        Z_final = z_trend + z_residual
        Z_final = torch.clamp(Z_final, min=-100.0, max=100.0)
        return Z_final


class SSM_Smoother(nn.Module):
    """SSM Smoother component from the reference implementation."""

    def __init__(
        self,
        feature_dim=4,
        keep_dim=2,
        time_dim=48,
        d_model=64,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        transition_g=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = n_head
        self.keep_dim = keep_dim
        self.time_dim = time_dim
        self.residual_mlp = nn.Sequential(
            nn.Linear(keep_dim * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, keep_dim),
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        self._cache = {}

    def forward(self, z_f_t, z_s_next, noise=None, noise_scale=0.1):
        B = z_f_t.shape[0]
        device = z_f_t.device
        if noise is None:
            noise = torch.randn(B, self.keep_dim, device=device)
        scaled_noise = noise * noise_scale
        fusion_input = torch.cat([z_f_t, z_s_next, scaled_noise], dim=-1)
        z_residual = self.residual_mlp(fusion_input)
        z_s_t = z_f_t + z_residual
        z_s_t = torch.clamp(z_s_t, min=-100.0, max=100.0)
        return z_s_t

    def precompute_history_features(self, H_full, time_stamps=None):
        self._cache = {"H_full": H_full}

    def clear_cache(self):
        self._cache = {}


class ResMLPBlock(nn.Module):
    """ResMLPBlock component from the reference implementation."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class SSM_Out_Fx_MLP(nn.Module):
    """SSM Out Fx MLP component from the reference implementation."""

    def __init__(self, feature_dim, keep_dim, d_model):
        super().__init__()
        self.noise_dim = feature_dim
        self.input_proj = nn.Linear(keep_dim + self.noise_dim, d_model)
        self.backbone = nn.Sequential(ResMLPBlock(d_model), ResMLPBlock(d_model), ResMLPBlock(d_model))
        self.head_pattern = nn.Linear(d_model, feature_dim)
        self.patient_adapter = nn.Sequential(
            nn.Linear(keep_dim, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, feature_dim * 2)
        )

    def forward(self, z_next, noise=None):
        B, T, _ = z_next.shape
        adapter_out = self.patient_adapter(z_next)
        p_scale, p_bias = adapter_out.chunk(2, dim=-1)
        p_scale = torch.clamp(F.softplus(p_scale) + 0.5, max=3.0)
        p_bias = torch.clamp(p_bias, min=-5.0, max=5.0)
        if noise is None:
            noise = torch.randn(B, T, self.noise_dim, device=z_next.device)
        combined = torch.cat([z_next, noise], dim=-1)
        h = self.backbone(self.input_proj(combined))
        raw_pattern = self.head_pattern(h)
        raw_pattern = torch.clamp(raw_pattern, -3.0, 3.0)
        x_gen = raw_pattern * p_scale + p_bias
        x_gen = torch.clamp(x_gen, min=-20.0, max=20.0)
        return x_gen


class SSM_Out_Fo_MLP(nn.Module):
    """SSM Out Fo MLP component from the reference implementation."""

    def __init__(self, feature_dim, keep_dim, d_model, zero_z=False, all_one=False):
        super().__init__()
        self.zero_z = zero_z
        self.all_one = all_one
        self.noise_dim = feature_dim
        self.input_proj = nn.Linear(keep_dim + self.noise_dim, d_model)
        self.backbone = nn.Sequential(ResMLPBlock(d_model), ResMLPBlock(d_model))
        self.head_pattern = nn.Linear(d_model, feature_dim)
        self.patient_adapter = nn.Sequential(
            nn.Linear(keep_dim, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, feature_dim * 2)
        )
        self.output_bias = nn.Parameter(torch.zeros(feature_dim))
        self._init_specific_weights()

    def _init_specific_weights(self):
        nn.init.constant_(self.patient_adapter[-1].bias, -2.0)
        nn.init.xavier_uniform_(self.head_pattern.weight, gain=0.1)
        nn.init.zeros_(self.head_pattern.bias)
        nn.init.zeros_(self.output_bias)

    def forward(self, z_next, noise=None):
        B, T, _ = z_next.shape
        if self.zero_z or self.all_one:
            z_next = torch.zeros_like(z_next)
        adapter_out = self.patient_adapter(z_next)
        logit_scale, logit_bias = adapter_out.chunk(2, dim=-1)
        logit_scale = F.softplus(logit_scale) + 0.1
        if noise is None:
            noise = torch.randn(B, T, self.noise_dim, device=z_next.device)
        combined = torch.cat([z_next, noise], dim=-1)
        h = self.backbone(self.input_proj(combined))
        raw_logits = self.head_pattern(h)
        final_logits = raw_logits * logit_scale + logit_bias + self.output_bias
        o_prob = torch.sigmoid(final_logits)
        if self.all_one:
            return torch.ones_like(o_prob)
        return o_prob


class SSM_Transition_G(nn.Module):
    """SSM Transition G component from the reference implementation."""

    def __init__(self, keep_dim, d_model, dropout=0.0):
        super().__init__()
        self.linear_A = nn.Linear(keep_dim, keep_dim, bias=False)
        nn.init.eye_(self.linear_A.weight)
        self.trend_net = nn.Sequential(
            nn.LayerNorm(keep_dim),
            nn.Linear(keep_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, keep_dim),
        )
        nn.init.constant_(self.trend_net[-1].weight, 0.0)
        nn.init.constant_(self.trend_net[-1].bias, 0.0)
        self.noise_fusion = nn.Sequential(
            nn.LayerNorm(keep_dim * 2), nn.Linear(keep_dim * 2, d_model), nn.GELU(), nn.Linear(d_model, keep_dim)
        )
        nn.init.normal_(self.noise_fusion[-1].weight, std=0.1)
        nn.init.constant_(self.noise_fusion[-1].bias, 0.0)
        self.output_norm = nn.LayerNorm(keep_dim)

    def forward(self, z_prev, noise=None):
        z_linear = self.linear_A(z_prev)
        z_trend_correction = self.trend_net(z_prev)
        if noise is None:
            noise = torch.randn_like(z_prev)
        fusion_input = torch.cat([z_prev, noise], dim=-1)
        z_stochastic = self.noise_fusion(fusion_input)
        z_next = self.output_norm(z_linear + z_trend_correction + z_stochastic)
        return z_next


class SSM_Decoder_V2(nn.Module):
    """SSM Decoder V2 component from the reference implementation."""

    def __init__(self, feature_dim, z_dim, d_model=64, n_heads=4, noise_dim=32, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.noise_dim = noise_dim
        self.z_to_base = nn.Linear(z_dim, feature_dim)
        self.res_proj = nn.Linear(1, d_model)
        self.z_ctx_proj = nn.Linear(z_dim, d_model)
        self.mask_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, feature_dim, d_model))
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_model * 2, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.adapter = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.pattern_proj = nn.Linear(d_model + noise_dim, d_model)
        self.backbone = nn.Sequential(ResMLPBlock(d_model, dropout), ResMLPBlock(d_model, dropout))
        self.head_pattern = nn.Linear(d_model, 1)

    def forward(self, z_t, x_obs, mask_obs, noise=None):
        B, T, p = x_obs.shape
        z_flat = z_t.reshape(B * T, -1)
        x_flat = x_obs.reshape(B * T, p)
        m_flat = mask_obs.reshape(B * T, p)
        x_base_guess = self.z_to_base(z_flat)
        residual = (x_flat - x_base_guess) * m_flat.float()
        emb_z = self.z_ctx_proj(z_flat).unsqueeze(1).expand(-1, p, -1)
        emb_res = self.res_proj(residual.unsqueeze(-1))
        emb_mask = self.mask_embedding(m_flat.long())
        emb_pos = self.position_embedding.expand(B * T, -1, -1)
        context = emb_res + emb_mask + emb_pos + emb_z
        context_norm = self.attn_norm(context)
        context_h, _ = self.feature_attention(context_norm, context_norm, context_norm)
        context = context + context_h
        context_norm = self.ffn_norm(context)
        context_ffn = self.ffn(context_norm)
        context = context + context_ffn
        adapter_out = self.adapter(context)
        p_scale, p_bias = adapter_out.chunk(2, dim=-1)
        p_scale = F.softplus(p_scale) + 0.5
        p_scale = p_scale.squeeze(-1)
        p_bias = p_bias.squeeze(-1)
        if noise is None:
            noise = torch.randn(B * T, p, self.noise_dim, device=z_t.device)
        gen_input = torch.cat([context, noise], dim=-1)
        h = self.backbone(self.pattern_proj(gen_input))
        raw_pattern = self.head_pattern(h).squeeze(-1)
        x_delta = raw_pattern * p_scale + p_bias
        x_gen = x_base_guess + x_delta
        x_gen = torch.clamp(x_gen, min=-100.0, max=100.0)
        return x_gen.reshape(B, T, p)
