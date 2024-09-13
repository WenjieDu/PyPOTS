"""

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn


class UsganDiscriminator(nn.Module):
    """model Discriminator: built on BiRNN

    Parameters
    ----------
    n_features :
        the feature dimension of the input

    rnn_hidden_size :
        the hidden size of the RNN cell

    hint_rate :
        the hint rate for the input imputed_data

    dropout_rate :
        the dropout rate for the output layer

    device :
        specify running the model on which device, CPU/GPU

    """

    def __init__(
        self,
        n_features: int,
        rnn_hidden_size: int,
        hint_rate: float = 0.7,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.hint_rate = hint_rate
        self.biRNN = nn.GRU(n_features * 2, rnn_hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.read_out = nn.Linear(rnn_hidden_size * 2, n_features)

    def forward(
        self,
        imputed_X: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward processing of USGAN Discriminator.

        Parameters
        ----------
        imputed_X : torch.Tensor,
            The original X with missing parts already imputed.

        missing_mask : torch.Tensor,
            The missing mask of X.

        Returns
        -------
        logits : torch.Tensor,
            the logits of the probability of being the true value.

        """

        device = imputed_X.device
        hint = torch.rand_like(missing_mask, dtype=torch.float, device=device) < self.hint_rate
        hint = hint.int()
        h = hint * missing_mask + (1 - hint) * 0.5
        x_in = torch.cat([imputed_X, h], dim=-1)

        out, _ = self.biRNN(x_in)
        logits = self.read_out(self.dropout(out))
        return logits
