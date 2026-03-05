"""
The backbone module of TKAN (Temporal Kolmogorov-Arnold Networks).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import List, Optional

import torch
import torch.nn as nn

from .layers import TKANCell


class BackboneTKAN(nn.Module):
    """Multi-layer TKAN backbone for sequence modelling.

    Stacks multiple :class:`TKANCell` layers to build a deep TKAN that
    processes a full time-series sequence and returns the output at every
    time step.

    Parameters
    ----------
    input_size :
        Dimensionality of the input at each time step.

    hidden_size :
        Dimensionality of the hidden state (same for every layer).

    n_layers :
        Number of stacked TKAN layers. Default: 1.

    sub_kan_configs :
        Configurations for the KAN sub-layers inside each cell. See
        :class:`TKANCell` for the accepted formats. Default: [None].

    sub_kan_output_dim :
        Output dimension of each sub-KAN layer inside a cell. Defaults to
        ``input_size`` for the first layer and ``hidden_size`` for subsequent
        layers unless explicitly provided.

    sub_kan_input_dim :
        Input dimension of each sub-KAN layer inside a cell. Behaves the same
        as ``sub_kan_output_dim``.

    dropout :
        Dropout probability applied to the output of each TKAN layer (except
        the last). Default: 0.0.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 1,
        sub_kan_configs: Optional[List] = None,
        sub_kan_output_dim: Optional[int] = None,
        sub_kan_input_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_rate = dropout

        cells = []
        for i in range(n_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            # For subsequent layers the sub-KAN dimensions should match hidden_size
            cell_sub_out = sub_kan_output_dim if sub_kan_output_dim is not None else cell_input_size
            cell_sub_in = sub_kan_input_dim if sub_kan_input_dim is not None else cell_input_size
            cells.append(
                TKANCell(
                    in_features=cell_input_size,
                    hidden_size=hidden_size,
                    sub_kan_configs=sub_kan_configs,
                    sub_kan_output_dim=cell_sub_out,
                    sub_kan_input_dim=cell_sub_in,
                )
            )
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over a full sequence.

        Parameters
        ----------
        x : shape [batch, seq_len, input_size]

        Returns
        -------
        outputs : shape [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        current = x  # [batch, seq_len, input_size or hidden_size]

        for layer_idx, cell in enumerate(self.cells):
            h, c, sub_states = cell.init_hidden_states(batch_size, x.device)
            step_outputs = []
            for t in range(seq_len):
                h, c, sub_states = cell(current[:, t, :], h, c, sub_states)
                step_outputs.append(h)

            # Stack: [seq_len, batch, hidden_size] -> [batch, seq_len, hidden_size]
            layer_out = torch.stack(step_outputs, dim=1)

            if self.dropout is not None and layer_idx < self.n_layers - 1:
                layer_out = self.dropout(layer_out)

            current = layer_out

        return current  # [batch, seq_len, hidden_size]
