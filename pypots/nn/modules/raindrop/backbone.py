"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class BackboneRaindrop(nn.Module):
    def __init__(
        self,
        n_features,
        n_layers,
        d_model,
        n_heads,
        d_ffn,
        n_classes,
        dropout=0.3,
        max_len=215,
        d_static=9,
        d_pe=16,
        aggregation="mean",
        sensor_wise_mask=False,
        static=False,
    ):

        try:
            from .layers import PositionalEncoding, ObservationPropagation
        except (ImportError, NameError) as e:
            raise ImportError(
                f"❌ {e}. Note that torch_geometric is missing, please install it with "
                "'pip install torch_geometric torch_scatter torch_sparse' or "
                "'conda install -c pyg pyg pytorch-scatter pytorch-sparse'"
            )

        super().__init__()

        self.n_layers = n_layers
        self.n_features = n_features
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.dropout = dropout
        self.max_len = max_len
        self.d_static = d_static
        self.aggregation = aggregation
        self.sensor_wise_mask = sensor_wise_mask
        self.static = static

        # create modules
        if self.static:
            self.static_emb = nn.Linear(d_static, n_features)
        else:
            self.static_emb = None
        assert d_model % n_features == 0, "d_model must be divisible by n_features"
        self.d_ob = int(d_model / n_features)
        self.encoder = nn.Linear(n_features * self.d_ob, n_features * self.d_ob)

        self.pos_encoder = PositionalEncoding(d_pe, max_len)
        if self.sensor_wise_mask:
            dim_check = n_features * (self.d_ob + d_pe)
            assert dim_check % n_heads == 0, "dim_check must be divisible by n_heads"
            encoder_layers = TransformerEncoderLayer(n_features * (self.d_ob + d_pe), n_heads, d_ffn, dropout)
        else:
            dim_check = d_model + d_pe
            assert dim_check % n_heads == 0, "dim_check must be divisible by n_heads"
            encoder_layers = TransformerEncoderLayer(d_model + d_pe, n_heads, d_ffn, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.R_u = nn.Parameter(torch.Tensor(1, self.n_features * self.d_ob))

        self.ob_propagation = ObservationPropagation(
            in_channels=max_len * self.d_ob,
            out_channels=max_len * self.d_ob,
            heads=1,
            n_nodes=n_features,
            ob_dim=self.d_ob,
        )
        self.ob_propagation_layer2 = ObservationPropagation(
            in_channels=max_len * self.d_ob,
            out_channels=max_len * self.d_ob,
            heads=1,
            n_nodes=n_features,
            ob_dim=self.d_ob,
        )
        if static:
            d_final = d_model + d_pe + n_features
        else:
            d_final = d_model + d_pe

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init_range = 1e-10
        self.encoder.weight.data.uniform_(-init_range, init_range)
        if self.static:
            self.static_emb.weight.data.uniform_(-init_range, init_range)
        nn.init.xavier_uniform(self.R_u)  # xavier_uniform also known as glorot

    def forward(
        self,
        X: torch.Tensor,
        timestamps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward processing of Raindrop.

        Parameters
        ----------
        X :
            The input tensor of shape (batch_size, n_features, max_len).

        timestamps :
            The timestamps tensor of shape (batch_size, max_len).

        lengths :
            The lengths tensor of shape (batch_size, 1).

        Returns
        -------
        prediction : torch.Tensor
        """
        src = X.permute(1, 0, 2)
        times = timestamps.permute(1, 0)

        device = src.device
        max_len, batch_size = src.shape[0], src.shape[1]

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src * self.R_u)
        pe = self.pos_encoder(times).to(device)
        h = self.dropout_layer(h)

        mask = torch.arange(max_len)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(device)

        x = h

        adj = torch.ones(self.n_features, self.n_features, device=device)
        adj[torch.eye(self.n_features, dtype=torch.bool)] = 1

        edge_index = torch.nonzero(adj).T
        edge_weights = adj[edge_index[0], edge_index[1]]

        output = torch.zeros([max_len, batch_size, self.n_features * self.d_ob], device=device)

        alpha_all = torch.zeros([edge_index.shape[1], batch_size], device=device)

        # iterate on each sample
        for unit in range(0, batch_size):
            step_data = x[:, unit, :]
            p_t = pe[:, unit, :]

            step_data = step_data.reshape([max_len, self.n_features, self.d_ob]).permute(1, 0, 2)
            step_data = step_data.reshape(self.n_features, max_len * self.d_ob)

            step_data, attention_weights = self.ob_propagation(
                step_data,
                p_t=p_t,
                edge_index=edge_index,
                edge_weights=edge_weights,
                use_beta=False,
                edge_attr=None,
                return_attention_weights=True,
            )

            edge_index_layer2 = attention_weights[0]
            edge_weights_layer2 = attention_weights[1].squeeze(-1)

            step_data, attention_weights = self.ob_propagation_layer2(
                step_data,
                p_t=p_t,
                edge_index=edge_index_layer2,
                edge_weights=edge_weights_layer2,
                use_beta=False,
                edge_attr=None,
                return_attention_weights=True,
            )

            step_data = step_data.view([self.n_features, max_len, self.d_ob])
            step_data = step_data.permute([1, 0, 2])  # [n_step, n_features, d_ob]
            step_data = step_data.reshape([-1, self.n_features * self.d_ob])

            output[:, unit, :] = step_data
            alpha_all[:, unit] = attention_weights[1].squeeze(-1)

        # distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
        # distance = torch.mean(distance)

        if self.sensor_wise_mask:
            extend_output = output.view(-1, batch_size, self.n_features, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.n_features, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.n_features * (self.d_ob + 16))
        else:
            output = torch.cat([output, pe], dim=2)

        output = self.transformer_encoder(output, src_key_padding_mask=mask)
        return output, mask
