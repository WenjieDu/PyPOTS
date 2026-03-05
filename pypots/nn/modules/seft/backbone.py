"""
The backbone of SeFT (Set Functions for Time Series).

Refer to the paper
`Max Horn, Michael Moor, Christian Bock, Bastian Rieck, and Karsten Borgwardt.
Set Functions for Time Series.
In the 37th International Conference on Machine Learning, 2020.
<https://proceedings.mlr.press/v119/horn20a.html>`_

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn


class _SAB(nn.Module):
    """Set Attention Block (SAB).

    Applies multi-head self-attention followed by a point-wise feed-forward
    network with residual connections and layer normalisation, as used in the
    Set Transformer / SeFT architecture.

    Parameters
    ----------
    d_model :
        Dimensionality of the set elements.

    n_heads :
        Number of attention heads.

    d_ffn :
        Hidden dimensionality of the position-wise feed-forward network.

    dropout :
        Dropout probability applied after attention and inside the FFN.
    """

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        X :
            Set representation of shape (batch_size, set_size, d_model).

        key_padding_mask :
            Boolean mask of shape (batch_size, set_size). ``True`` positions
            are ignored (i.e. correspond to missing / padding elements).

        Returns
        -------
        torch.Tensor
            Transformed set of shape (batch_size, set_size, d_model).
        """
        attn_out, _ = self.attn(X, X, X, key_padding_mask=key_padding_mask)
        X = self.norm1(X + self.dropout(attn_out))
        ffn_out = self.ffn(X)
        X = self.norm2(X + self.dropout(ffn_out))
        return X


class _PMA(nn.Module):
    """Pooling by Multi-head Attention (PMA).

    Aggregates a variable-size set into a fixed number of seed vectors via
    cross-attention, as proposed in the Set Transformer paper and used in SeFT.

    Parameters
    ----------
    d_model :
        Dimensionality of the set elements and seed vectors.

    n_heads :
        Number of attention heads.

    n_seeds :
        Number of seed vectors (output size of the aggregation).

    d_ffn :
        Hidden dimensionality of the feed-forward network applied to the seeds.

    dropout :
        Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, n_seeds: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(1, n_seeds, d_model))
        nn.init.xavier_uniform_(self.seeds)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        X :
            Set representation of shape (batch_size, set_size, d_model).

        key_padding_mask :
            Boolean mask of shape (batch_size, set_size). ``True`` positions
            are treated as padding (missing values).

        Returns
        -------
        torch.Tensor
            Aggregated representation of shape (batch_size, n_seeds, d_model).
        """
        batch_size = X.size(0)
        # expand seed vectors to the batch dimension
        Q = self.seeds.expand(batch_size, -1, -1)
        attn_out, _ = self.attn(Q, X, X, key_padding_mask=key_padding_mask)
        Q = self.norm1(Q + self.dropout(attn_out))
        ffn_out = self.ffn(Q)
        Q = self.norm2(Q + self.dropout(ffn_out))
        return Q


class BackboneSeFT(nn.Module):
    """Backbone of SeFT (Set Functions for Time Series).

    Represents each time series as a *set* of observation triples
    ``(normalised_time, feature_index, value)`` and processes them with
    multiple Set Attention Blocks (SABs) followed by Pooling by Multi-head
    Attention (PMA).

    Parameters
    ----------
    n_steps :
        Number of time steps in the input time series.

    n_features :
        Number of features (channels) in the time series.

    n_classes :
        Number of output classes.

    n_layers :
        Number of SAB (self-attention) layers.

    n_heads :
        Number of attention heads in each SAB and in PMA.

    d_model :
        Dimensionality used throughout the set-function encoder.

    d_ffn :
        Hidden dimensionality of the point-wise feed-forward networks.

    n_seeds :
        Number of seed vectors for the PMA pooling step.

    dropout :
        Dropout probability applied inside SAB and PMA.

    max_timescale :
        The maximum timescale for the sinusoidal time encoding.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ffn: int,
        n_seeds: int = 4,
        dropout: float = 0.0,
        max_timescale: float = 100.0,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_model = d_model
        self.n_seeds = n_seeds
        self.max_timescale = max_timescale

        # The time encoding occupies d_model // 2 dimensions (sin + cos components).
        # It must be even since we concatenate sin and cos parts of equal size.
        d_time = max((d_model // 2) // 2 * 2, 2)
        # The feature embedding occupies d_model // 4 dimensions (at least 4)
        d_feat = max(d_model // 4, 4)
        # +1 for the scalar observation value
        d_obs = d_time + d_feat + 1

        self.d_time = d_time
        self.d_feat = d_feat

        self.feature_embedding = nn.Embedding(n_features, d_feat)
        self.input_projection = nn.Linear(d_obs, d_model)

        self.sab_layers = nn.ModuleList(
            [_SAB(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )
        self.pma = _PMA(d_model, n_heads, n_seeds, d_ffn, dropout)
        self.classifier = nn.Linear(n_seeds * d_model, n_classes)

    def _time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time encoding.

        Parameters
        ----------
        t :
            Normalised timestamps of shape (batch_size, n_steps) in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Encoding of shape (batch_size, n_steps, d_time).
        """
        d = self.d_time
        i = torch.arange(0, d // 2, device=t.device, dtype=t.dtype)
        freq = 1.0 / (self.max_timescale ** (2.0 * i / d))
        # angles: (batch_size, n_steps, d//2)
        angles = t.unsqueeze(-1) * freq.view(1, 1, -1)
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return enc

    def forward(self, X: torch.Tensor, missing_mask: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        X :
            Time-series data of shape (batch_size, n_steps, n_features).
            Missing positions should have been zero-filled before passing here.

        missing_mask :
            Binary mask of shape (batch_size, n_steps, n_features).
            ``1`` indicates an *observed* value, ``0`` indicates a *missing* value.

        Returns
        -------
        torch.Tensor
            Class logits of shape (batch_size, n_classes).
        """
        batch_size, n_steps, n_features = X.shape
        device = X.device

        # ── time encoding ─────────────────────────────────────────────────
        t = torch.linspace(0.0, 1.0, n_steps, device=device)
        t = t.unsqueeze(0).expand(batch_size, -1)  # (B, T)
        time_enc = self._time_encoding(t)  # (B, T, d_time)
        # expand over features: (B, T, F, d_time)
        time_enc = time_enc.unsqueeze(2).expand(-1, -1, n_features, -1)

        # ── feature embedding ─────────────────────────────────────────────
        feat_idx = torch.arange(n_features, device=device)
        feat_emb = self.feature_embedding(feat_idx)  # (F, d_feat)
        # expand over batch and time: (B, T, F, d_feat)
        feat_emb = feat_emb.view(1, 1, n_features, self.d_feat).expand(batch_size, n_steps, -1, -1)

        # ── concatenate observation representation ────────────────────────
        values = X.unsqueeze(-1)  # (B, T, F, 1)
        obs_repr = torch.cat([time_enc, feat_emb, values], dim=-1)  # (B, T, F, d_obs)

        # ── flatten to set ────────────────────────────────────────────────
        set_size = n_steps * n_features
        obs_repr = obs_repr.reshape(batch_size, set_size, -1)  # (B, S, d_obs)
        set_repr = self.input_projection(obs_repr)  # (B, S, d_model)

        # ── attention key-padding mask ────────────────────────────────────
        # True  → element is *missing* (padding), ignore in attention
        obs_mask = missing_mask.reshape(batch_size, set_size)  # (B, S)
        key_padding_mask = obs_mask == 0  # True for missing

        # ── SAB layers ────────────────────────────────────────────────────
        for sab in self.sab_layers:
            set_repr = sab(set_repr, key_padding_mask=key_padding_mask)

        # ── PMA aggregation ───────────────────────────────────────────────
        pooled = self.pma(set_repr, key_padding_mask=key_padding_mask)  # (B, n_seeds, d_model)

        # ── classification head ───────────────────────────────────────────
        pooled_flat = pooled.reshape(batch_size, -1)  # (B, n_seeds * d_model)
        logits = self.classifier(pooled_flat)  # (B, n_classes)
        return logits
