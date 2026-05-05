"""
The package including the modules of HELIX.

Refer to the paper
`Fengming Zhang, Wenjie Du, Huan Zhang, Ke Yu, and Shen Qu.
HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation.
ICML (spotlight), 2026.
<https://openreview.net/forum?id=FN20iuPnEU>`_

Notes
-----
Refer to the repo https://github.com/milaogou/HELIX for details.

"""

# Created by Fengming Zhang <milaogou@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ..transformer.embedding import PositionalEncoding


class TimeSeriesEmbedding2D(nn.Module):
    """Embedding layer for 2D time series data [B, T, F]."""

    def __init__(self, n_features, pe_dim=16, feature_embed_dim=1):
        super().__init__()
        self.n_features = n_features
        self.pe_dim = pe_dim
        self.feature_embed_dim = feature_embed_dim

        # Rotary positional encoding for temporal dimension
        self.temporal_pe = PositionalEncoding(d_hid=pe_dim)

        # Learnable identity embedding for feature dimension
        self.feature_id = nn.Parameter(torch.randn(n_features, feature_embed_dim))

    def forward(self, X, missing_mask):
        """
        Parameters
        ----------
        X : tensor, shape [B, T, F]
            Input data with missing values filled
        missing_mask : tensor, shape [B, T, F]
            Mask indicating observed values (1) and missing values (0)

        Returns
        -------
        embedded : tensor, shape [B, T, F, embed_dim]
            Embedded data, embed_dim = 1 + pe_dim + feature_embed_dim + 1
        """
        B, T, F = X.shape

        # Data value [B, T, F, 1]
        data_val = X.unsqueeze(-1)

        # Temporal positional encoding
        temporal_encoding = self.temporal_pe(X, return_only_pos=True)  # [T, pe_dim]
        temporal_encoding = temporal_encoding.unsqueeze(2)  # [B, T, 1, pe_dim]
        temporal_encoding = temporal_encoding.expand(B, T, F, self.pe_dim)

        # Feature identity embedding [F, feature_embed_dim] -> [B, T, F, feature_embed_dim]
        feature_embedding = self.feature_id.unsqueeze(0).unsqueeze(0)  # [1, 1, F, feature_embed_dim]
        feature_embedding = feature_embedding.expand(B, T, F, self.feature_embed_dim)

        # Missing mask [B, T, F, 1]
        mask_feature = missing_mask.unsqueeze(-1)

        # Concatenate all embeddings
        embedded = torch.cat(
            [
                data_val,  # [B, T, F, 1]
                temporal_encoding,  # [B, T, F, pe_dim]
                feature_embedding,  # [B, T, F, feature_embed_dim]
                mask_feature,  # [B, T, F, 1]
            ],
            dim=-1,
        )  # [B, T, F, pe_dim + feature_embed_dim + 2]

        return embedded


class FeatureProjection(nn.Module):
    """Project between embedding dimension and model dimension."""

    def __init__(self, input_dim, d_model):
        super().__init__()
        self.forward_proj = nn.Linear(input_dim, d_model)
        self.backward_proj = nn.Linear(d_model, input_dim)

    def project_forward(self, x):
        return self.forward_proj(x)

    def project_backward(self, x):
        return self.backward_proj(x)


class UnifiedAttentionEncoder(nn.Module):
    """Unified attention encoder that can be applied to any dimension.

    Modified to save attention weights for visualization.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Storage for attention weights (for visualization)
        self.last_attn_weights = None

    def forward(self, x):
        # Self-attention with weight capture
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)

        # Save attention weights (detached to avoid affecting gradients)
        self.last_attn_weights = attn_weights.detach()

        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


class DimensionalAttention(nn.Module):
    """Apply attention along a specific dimension.

    Modified to provide access to attention weights.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.encoder = UnifiedAttentionEncoder(d_model, n_heads, dropout)

    def get_last_attn_weights(self):
        """Get the last attention weights from the encoder."""
        return self.encoder.last_attn_weights

    def forward(self, x, target_dim):
        """
        Parameters
        ----------
        x : tensor, shape [B, T, F, D]
            Input tensor
        target_dim : str
            'time' or 'feature'

        Returns
        -------
        output : tensor, shape [B, T, F, D]
            Output after applying attention along target dimension
        """
        B, T, F, D = x.shape

        if target_dim == "time":
            # Apply attention along time dimension
            # Reshape to [B*F, T, D]
            x_reshaped = x.permute(0, 2, 1, 3).reshape(B * F, T, D)
            out = self.encoder(x_reshaped)
            # Reshape back to [B, T, F, D]
            out = out.reshape(B, F, T, D).permute(0, 2, 1, 3)

        elif target_dim == "feature":
            # Apply attention along feature dimension
            # Reshape to [B*T, F, D]
            x_reshaped = x.reshape(B * T, F, D)
            out = self.encoder(x_reshaped)
            # Reshape back to [B, T, F, D]
            out = out.reshape(B, T, F, D)
        else:
            raise ValueError(f"Invalid target_dim: {target_dim}")

        return out


class BackboneHELIX(nn.Module):
    """
    HELIX-2D backbone with hybrid parallel and serial cross-dimensional encoding.

    Modified to provide access to all attention weights.
    """

    def __init__(
        self,
        n_features,
        d_pe,
        d_feature_embed,
        d_model,
        n_heads,
        n_layers,
        dropout,
    ):
        super().__init__()

        # Embedding
        embed_dim = d_pe + d_feature_embed + 2  # 1(data) + pe_dim(temporal) + feature_embed_dim + 1(mask)
        self.embedding = TimeSeriesEmbedding2D(n_features, d_pe, d_feature_embed)

        # Projection
        self.projection = FeatureProjection(embed_dim, d_model)

        # Multi-layer encoders
        self.encoders = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "time": DimensionalAttention(d_model, n_heads, dropout),
                        "feature": DimensionalAttention(d_model, n_heads, dropout),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

        # Store number of layers for weight access
        self.n_layers = n_layers

    def get_attention_weights(self):
        """
        Get all attention weights from all layers.

        Returns
        -------
        attention_dict : dict
            Dictionary containing attention weights for each layer and dimension.
            Keys: 'layer{i}_time', 'layer{i}_feature' for i in range(n_layers)
            Values: attention weight tensors
        """
        attention_dict = {}
        for i, layer_encoders in enumerate(self.encoders):
            time_attn = layer_encoders["time"].get_last_attn_weights()
            feature_attn = layer_encoders["feature"].get_last_attn_weights()

            if time_attn is not None:
                attention_dict[f"layer{i}_time"] = time_attn
            if feature_attn is not None:
                attention_dict[f"layer{i}_feature"] = feature_attn

        return attention_dict

    def forward(self, X, missing_mask):
        """
        Parameters
        ----------
        X : tensor, shape [B, T, F]
            Input data
        missing_mask : tensor, shape [B, T, F]
            Missing mask

        Returns
        -------
        reconstruction : tensor, shape [B, T, F]
            Reconstructed data
        """
        # Embedding
        embedded = self.embedding(X, missing_mask)  # [B, T, F, embed_dim]

        # Project to model dimension
        x = self.projection.project_forward(embedded)  # [B, T, F, d_model]

        # Store all intermediate outputs for multi-level fusion
        all_outputs = [x]

        # Multi-layer hybrid encoding
        for layer_encoders in self.encoders:
            # Phase 1: Parallel encoding
            time_encoded = layer_encoders["time"](x, "time")
            feat_encoded = layer_encoders["feature"](x, "feature")

            all_outputs.extend([time_encoded, feat_encoded])

            # Phase 2: Cross-dimensional serial encoding
            time_feat = layer_encoders["feature"](time_encoded, "feature")
            feat_time = layer_encoders["time"](feat_encoded, "time")

            all_outputs.extend([time_feat, feat_time])

            # Intra-layer fusion: average of 4 outputs for next layer
            x = torch.stack([time_encoded, feat_encoded, time_feat, feat_time], dim=0).mean(dim=0)

        # Global fusion: average of all intermediate outputs
        fused = torch.stack(all_outputs, dim=0).mean(dim=0)
        fused = self.final_norm(fused)

        # Project back to 1D
        output = self.output_proj(fused).squeeze(-1)  # [B, T, F]

        return output
