"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn


class PITSBlock(nn.Module):
    """A single block that processes each patch independently (no cross-patch attention)."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [bz * nvars, n_patches, d_model]
        # Process each patch position independently (shared weights, no cross-patch attention)
        return x + self.dropout(self.ff(self.norm(x)))


class PITSBackbone(nn.Module):
    """Patch-independent backbone for PITS.

    Unlike PatchTST which uses self-attention between patches, PITS processes
    each patch independently via a simple MLP, with no information exchange
    between patches.
    """

    def __init__(self, n_layers: int, d_model: int, d_ffn: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([PITSBlock(d_model, d_ffn, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # x: [bz * nvars, n_patches, d_model]
        for block in self.blocks:
            x = block(x)
        # Reshape to match PredictionHead expectation: [bz, nvars, d_model, n_patches]
        enc_out = x.reshape(-1, self.d_model, x.shape[-2], x.shape[-1])  # [bz, nvars, n_patches, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [bz, nvars, d_model, n_patches]
        return enc_out
