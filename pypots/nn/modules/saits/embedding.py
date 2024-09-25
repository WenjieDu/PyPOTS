"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ..transformer import PositionalEncoding


class SaitsEmbedding(nn.Module):
    """The embedding method from the SAITS paper :cite:`du2023SAITS`.

    Parameters
    ----------
    d_in :
        The input dimension.

    d_out :
        The output dimension.

    with_pos :
        Whether to add positional encoding.

    n_max_steps :
        The maximum number of steps.
        It only works when ``with_pos`` is True.

    dropout :
        The dropout rate.

    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        with_pos: bool,
        n_max_steps: int = 1000,
        dropout: float = 0,
    ):
        super().__init__()
        self.with_pos = with_pos
        self.dropout_rate = dropout

        self.embedding_layer = nn.Linear(d_in, d_out)
        self.position_enc = PositionalEncoding(d_out, n_positions=n_max_steps) if with_pos else None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, X, missing_mask=None):
        if missing_mask is not None:
            X = torch.cat([X, missing_mask], dim=2)

        X_embedding = self.embedding_layer(X)

        if self.with_pos:
            X_embedding = self.position_enc(X_embedding)
        if self.dropout_rate > 0:
            X_embedding = self.dropout(X_embedding)

        return X_embedding
