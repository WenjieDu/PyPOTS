"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class ImplicitImputation(nn.Module):
    def __init__(self, d_input: int):
        super().__init__()
        self.projection_layer = nn.Linear(d_input, d_input, bias=False)

    def forward(self, X: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        imputation = self.projection_layer(X)
        imputed_X = X * missing_mask + imputation * (1 - X)
        return imputed_X


class PeepholeLSTMCell(nn.LSTMCell):
    """
    Notes
    -----
    This implementation is adapted from https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81

    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias)
        self.weight_ch = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ch = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter("bias_ch", None)
        self.register_buffer("wc_blank", torch.zeros(hidden_size))
        self.reset_parameters()

    def forward(
        self,
        X: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            zeros = torch.zeros(X.size(0), self.hidden_size, dtype=X.dtype, device=X.device)
            hx = (zeros, zeros)

        h, c = hx

        wx = F.linear(X, self.weight_ih, self.bias_ih)
        wh = F.linear(h, self.weight_hh, self.bias_hh)
        wc = F.linear(c, self.weight_ch, self.bias_ch)

        wxhc = (
            wx
            + wh
            + torch.cat(
                (
                    wc[:, : 2 * self.hidden_size],
                    Variable(self.wc_blank).expand_as(h),
                    wc[:, 2 * self.hidden_size :],
                ),
                dim=1,
            )
        )

        i = torch.sigmoid(wxhc[:, : self.hidden_size])
        f = torch.sigmoid(wxhc[:, self.hidden_size : 2 * self.hidden_size])
        g = torch.tanh(wxhc[:, 2 * self.hidden_size : 3 * self.hidden_size])
        o = torch.sigmoid(wxhc[:, 3 * self.hidden_size :])

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class GMMLayer(nn.Module):
    def __init__(self, d_hidden: int, n_clusters: int):
        super().__init__()
        self.mu_c_unscaled = Parameter(torch.Tensor(n_clusters, d_hidden))
        self.var_c_unscaled = Parameter(torch.Tensor(n_clusters, d_hidden))
        self.phi_c_unscaled = Parameter(torch.Tensor(n_clusters))

    def set_values(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
        phi: torch.Tensor,
    ) -> None:
        assert mu.shape == self.mu_c_unscaled.shape
        assert var.shape == self.var_c_unscaled.shape
        assert phi.shape == self.phi_c_unscaled.shape
        self.mu_c_unscaled = torch.nn.Parameter(mu)
        self.var_c_unscaled = torch.nn.Parameter(var)
        self.phi_c_unscaled = torch.nn.Parameter(phi)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_c = self.mu_c_unscaled
        var_c = F.softplus(self.var_c_unscaled)
        phi_c = torch.softmax(self.phi_c_unscaled, dim=0)
        return mu_c, var_c, phi_c
