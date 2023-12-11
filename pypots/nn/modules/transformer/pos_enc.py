"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """The original positional-encoding module for Transformer.

    Parameters
    ----------
    d_hid:
        The dimension of the hidden layer.

    n_positions:
        The max number of positions.

    """

    def __init__(self, d_hid: int, n_positions: int = 1000):
        super().__init__()
        pe = torch.zeros(n_positions, d_hid, requires_grad=False).float()
        position = torch.arange(0, n_positions).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_hid, 2).float()
            * -(torch.log(torch.tensor(10000)) / d_hid)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pos_table", pe)

    def forward(self, x: torch.Tensor, return_only_pos: bool = False) -> torch.Tensor:
        """Forward processing of the positional encoding module.

        Parameters
        ----------
        x:
            Input tensor.

        return_only_pos:
            Whether to return only the positional encoding.

        Returns
        -------
        If return_only_pos is True:
            pos_enc:
                The positional encoding.
        else:
            x_with_pos:
                Output tensor, the input tensor with the positional encoding added.
        """
        pos_enc = self.pos_table[:, : x.size(1)].clone().detach()

        if return_only_pos:
            return pos_enc

        x_with_pos = x + pos_enc
        return x_with_pos
