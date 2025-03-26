"""
The core wrapper assembles the submodules of ImputeFormer imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Tong Nie <nietong@tongji.edu.cn> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.imputeformer import (
    EmbeddedAttentionLayer,
    ProjectedAttentionLayer,
    MLP,
)
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import SaitsLoss


class _ImputeFormer(ModelCore):
    """
    Spatiotemporal Imputation Transformer induced by low-rank factorization, KDD'24.
    Note:
        This is a simplified implementation under the SAITS framework (ORT+MIT).
        The timestamp encoding is also removed for ease of implementation.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_input_embed: int,
        d_learnable_embed: int,
        d_proj: int,
        d_ffn: int,
        n_temporal_heads: int,
        dropout: float,
        input_dim: int,
        output_dim: int,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_nodes = n_features
        self.in_steps = n_steps
        self.out_steps = n_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = d_input_embed
        self.learnable_embedding_dim = d_learnable_embed
        self.model_dim = d_input_embed + d_learnable_embed

        self.n_temporal_heads = n_temporal_heads
        self.num_layers = n_layers
        self.input_proj = nn.Linear(input_dim, self.input_embedding_dim)
        self.d_proj = d_proj
        self.d_ffn = d_ffn

        self.learnable_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.in_steps, self.n_nodes, self.learnable_embedding_dim))
        )

        self.readout = MLP(self.model_dim, self.model_dim, output_dim, n_layers=2)

        self.attn_layers_t = nn.ModuleList(
            [
                ProjectedAttentionLayer(
                    self.n_nodes,
                    self.d_proj,
                    self.model_dim,
                    self.n_temporal_heads,
                    self.model_dim,
                    dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                EmbeddedAttentionLayer(
                    self.model_dim,
                    self.learnable_embedding_dim,
                    self.d_ffn,
                )
                for _ in range(self.num_layers)
            ]
        )

        # apply SAITS loss function to Imputeformer on the imputation task
        self.training_loss = SaitsLoss(ORT_weight, MIT_weight, training_loss)
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        x, missing_mask = inputs["X"], inputs["missing_mask"]

        # x: (batch_size, in_steps, num_nodes)
        # Note that ImputeFormer is designed for Spatial-Temporal data that has the format [B, S, N, C],
        # where N is the number of nodes and C is an additional feature dimension,
        # We simply add an extra axis here for implementation.
        x = x.unsqueeze(-1)  # [b s n c]
        missing_mask = missing_mask.unsqueeze(-1)  # [b s n c]
        batch_size = x.shape[0]
        # Whiten missing values
        x = x * missing_mask
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)

        # Learnable node embedding
        node_emb = self.learnable_embedding.expand(batch_size, *self.learnable_embedding.shape)
        x = torch.cat([x, node_emb], dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        # Spatial and temporal processing with customized attention layers
        x = x.permute(0, 2, 1, 3)  # [b n s c]
        for att_t, att_s in zip(self.attn_layers_t, self.attn_layers_s):
            x = att_t(x)
            x = att_s(x, self.learnable_embedding, dim=1)

        # Readout
        x = x.permute(0, 2, 1, 3)  # [b s n c]
        reconstruction = self.readout(x)
        reconstruction = reconstruction.squeeze(-1)  # [b s n]
        missing_mask = missing_mask.squeeze(-1)  # [b s n]

        # Below is the SAITS processing pipeline:
        # replace the observed part with values from X
        imputed_data = missing_mask * inputs["X"] + (1 - missing_mask) * reconstruction

        # ensemble the results as a dictionary for return
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                loss, ORT_loss, MIT_loss = self.training_loss(reconstruction, X_ori, missing_mask, indicating_mask)
                results["ORT_loss"] = ORT_loss
                results["MIT_loss"] = MIT_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
