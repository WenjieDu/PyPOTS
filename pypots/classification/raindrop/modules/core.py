"""
The implementation of Raindrop for the partially-observed time-series classification task.

Refer to the paper "Zhang, X., Zeman, M., Tsiligkaridis, T., & Zitnik, M. (2022).
Graph-Guided Network for Irregularly Sampled Multivariate Time Series. ICLR 2022."

"""


# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.parameter import Parameter

from ....utils.logging import logger

try:
    from .submodules import PositionalEncoding, ObservationPropagation
    from torch_geometric.nn.inits import glorot
except ImportError as e:
    logger.error(
        f"❌ {e}\n"
        "Note torch_geometric is missing, please install it with "
        "'pip install torch_geometric torch_scatter torch_sparse' or "
        "'conda install -c pyg pyg pytorch-scatter pytorch-sparse'"
    )
except NameError as e:
    logger.error(
        f"❌ {e}\n"
        "Note torch_geometric is missing, please install it with "
        "'pip install torch_geometric torch_scatter torch_sparse' or "
        "'conda install -c pyg pyg pytorch-scatter pytorch-sparse'"
    )


class _Raindrop(nn.Module):
    def __init__(
        self,
        n_features,
        n_layers,
        d_model,
        d_ffn,
        n_heads,
        n_classes,
        dropout=0.3,
        max_len=215,
        d_static=9,
        aggregation="mean",
        sensor_wise_mask=False,
        static=False,
        device=None,
    ):
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
        self.device = device

        # create modules
        if self.static:
            self.static_emb = nn.Linear(d_static, n_features)
        else:
            self.static_emb = None
        assert d_model % n_features == 0, "d_model must be divisible by n_features"
        self.d_ob = int(d_model / n_features)
        self.encoder = nn.Linear(n_features * self.d_ob, n_features * self.d_ob)
        d_pe = 16
        self.pos_encoder = PositionalEncoding(d_pe, max_len)
        if self.sensor_wise_mask:
            dim_check = n_features * (self.d_ob + d_pe)
            assert dim_check % n_heads == 0, "dim_check must be divisible by n_heads"
            encoder_layers = TransformerEncoderLayer(
                n_features * (self.d_ob + d_pe), n_heads, d_ffn, dropout
            )
        else:
            dim_check = d_model + d_pe
            assert dim_check % n_heads == 0, "dim_check must be divisible by n_heads"
            encoder_layers = TransformerEncoderLayer(
                d_model + d_pe, n_heads, d_ffn, dropout
            )
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.R_u = Parameter(torch.Tensor(1, self.n_features * self.d_ob))

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

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init_range = 1e-10
        self.encoder.weight.data.uniform_(-init_range, init_range)
        if self.static:
            self.static_emb.weight.data.uniform_(-init_range, init_range)
        glorot(self.R_u)

    def classify(self, inputs: dict) -> torch.Tensor:
        """Forward processing of BRITS.

        Parameters
        ----------
        inputs : dict,
            The input data.

        Returns
        -------
        prediction : torch.Tensor
        """
        src = inputs["X"].permute(1, 0, 2)
        static = inputs["static"]
        times = inputs["timestamps"].permute(1, 0)
        lengths = inputs["lengths"]
        missing_mask = inputs["missing_mask"].permute(1, 0, 2)

        max_len, batch_size = src.shape[0], src.shape[1]

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src * self.R_u)
        pe = self.pos_encoder(times).to(src.device)
        if static is not None:
            emb = self.static_emb(static)

        h = self.dropout(h)

        mask = torch.arange(max_len)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(src.device)

        x = h

        adj = torch.ones(self.n_features, self.n_features, device=src.device)
        adj[torch.eye(self.n_features, dtype=torch.bool)] = 1

        edge_index = torch.nonzero(adj).T
        edge_weights = adj[edge_index[0], edge_index[1]]

        batch_size = src.shape[1]
        n_step = src.shape[0]
        output = torch.zeros(
            [n_step, batch_size, self.n_features * self.d_ob], device=src.device
        )

        alpha_all = torch.zeros([edge_index.shape[1], batch_size], device=src.device)

        # iterate on each sample
        for unit in range(0, batch_size):
            step_data = x[:, unit, :]
            p_t = pe[:, unit, :]

            step_data = step_data.reshape([n_step, self.n_features, self.d_ob]).permute(
                1, 0, 2
            )
            step_data = step_data.reshape(self.n_features, n_step * self.d_ob)

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

            step_data = step_data.view([self.n_features, n_step, self.d_ob])
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

        r_out = self.transformer_encoder(output, src_key_padding_mask=mask)

        lengths2 = lengths.unsqueeze(1).to(src.device)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        if self.sensor_wise_mask:
            output = torch.zeros(
                [batch_size, self.n_features, self.d_ob + 16], device=src.device
            )
            extended_missing_mask = missing_mask.view(-1, batch_size, self.n_features)
            for se in range(self.n_features):
                r_out = r_out.view(-1, batch_size, self.n_features, (self.d_ob + 16))
                out = r_out[:, :, se, :]
                l_ = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(
                    1
                )  # length
                out_sensor = torch.sum(
                    out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0
                ) / (l_ + 1)
                output[:, se, :] = out_sensor
            output = output.view([-1, self.n_features * (self.d_ob + 16)])
        elif self.aggregation == "mean":
            output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        else:
            raise RuntimeError

        if static is not None:
            output = torch.cat([output, emb], dim=1)

        logits = self.mlp_static(output)
        prediction = torch.softmax(logits, dim=1)

        return prediction

    def forward(self, inputs, training=True):
        classification_pred = self.classify(inputs)
        results = {"classification_pred": classification_pred}

        # if in training mode, return results with losses
        if training:
            classification_loss = F.nll_loss(
                torch.log(classification_pred), inputs["label"]
            )
            results["loss"] = classification_loss

        return results
