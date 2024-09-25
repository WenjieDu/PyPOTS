"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class CsdiDiffusionEmbedding(nn.Module):
    def __init__(self, n_diffusion_steps, d_embedding=128, d_projection=None):
        super().__init__()
        if d_projection is None:
            d_projection = d_embedding
        self.register_buffer(
            "embedding",
            self._build_embedding(n_diffusion_steps, d_embedding // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(d_embedding, d_projection)
        self.projection2 = nn.Linear(d_projection, d_projection)

    @staticmethod
    def _build_embedding(n_steps, d_embedding=64):
        steps = torch.arange(n_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(d_embedding) / (d_embedding - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

    def forward(self, diffusion_step: int):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


class CsdiResidualBlock(nn.Module):
    def __init__(self, d_side, n_channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, n_channels)
        self.cond_projection = conv1d_with_init(d_side, 2 * n_channels, 1)
        self.mid_projection = conv1d_with_init(n_channels, 2 * n_channels, 1)
        self.output_projection = conv1d_with_init(n_channels, 2 * n_channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=n_channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=n_channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape  # bz, 2, n_features, n_steps
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape  # bz, 2, n_features, n_steps
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)

        y = x + diffusion_emb
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class CsdiDiffusionModel(nn.Module):
    def __init__(
        self,
        n_diffusion_steps,
        d_diffusion_embedding,
        d_input,
        d_side,
        n_channels,
        n_heads,
        n_layers,
    ):
        super().__init__()
        self.diffusion_embedding = CsdiDiffusionEmbedding(
            n_diffusion_steps=n_diffusion_steps,
            d_embedding=d_diffusion_embedding,
        )
        self.input_projection = conv1d_with_init(d_input, n_channels, 1)
        self.output_projection1 = conv1d_with_init(n_channels, n_channels, 1)
        self.output_projection2 = conv1d_with_init(n_channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                CsdiResidualBlock(
                    d_side=d_side,
                    n_channels=n_channels,
                    diffusion_embedding_dim=d_diffusion_embedding,
                    nheads=n_heads,
                )
                for _ in range(n_layers)
            ]
        )
        self.n_channels = n_channels

    def forward(self, x, cond_info, diffusion_step):
        (
            n_samples,
            input_dim,
            n_features,
            n_steps,
        ) = x.shape  # n_samples, 2, n_features, n_steps

        x = x.reshape(n_samples, input_dim, n_features * n_steps)
        x = self.input_projection(x)  # n_samples, n_channels, n_features*n_steps
        x = F.relu(x)
        x = x.reshape(n_samples, self.n_channels, n_features, n_steps)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(n_samples, self.n_channels, n_features * n_steps)
        x = self.output_projection1(x)  # (n_samples, channel, n_features*n_steps)
        x = F.relu(x)
        x = self.output_projection2(x)  # (n_samples, 1, n_features*n_steps)
        x = x.reshape(n_samples, n_features, n_steps)
        return x
