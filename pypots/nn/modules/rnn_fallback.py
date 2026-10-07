"""
Device-adaptive RNN layers: fused nn.GRU/LSTM on CUDA/CPU, equivalent unfused
step-loop implementations on Ascend NPU.

Why: on Ascend NPU (CANN 8.x/9.x), the fused kernel behind full-sequence
nn.GRU/LSTM (DynamicGRUV2) has no fp32 backward pass — fp32 training fails
asynchronously (usually surfacing in optimizer.step). The step-loops below
compose only base aten ops (matmul/sigmoid/tanh), so fp32 forward+backward
both run on NPU, with no precision loss and no AMP dependency.

Equivalence: parameters are registered under the exact nn.GRU/nn.LSTM names
(weight_ih_l0 / weight_hh_l0 / bias_ih_l0 / bias_hh_l0, *_reverse for the
backward direction), so state_dicts interchange directly at any nesting depth —
a checkpoint saved from a fused model loads into the unfused one and vice versa.
The per-step math is identical to GRUCell/LSTMCell, hence to the fused kernels.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk>
# License: BSD-3-Clause

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _use_unfused_rnn() -> bool:
    """Whether to pick the unfused step-loop implementations: only on NPU."""
    try:
        import torch_npu

        return torch_npu.npu.is_available()
    except ImportError:
        return False


class UnfusedGRU(nn.Module):
    """Unfused full-sequence GRU: manual step-loop, drop-in for nn.GRU.

    Supports the subset of nn.GRU options used by PyPOTS models:
    single layer, optional bidirectional, batch_first.
    Parameter names match nn.GRU so state_dicts interchange at any nesting depth.
    """

    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        gate_size = 3 * hidden_size
        self.weight_ih_l0 = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(gate_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(gate_size))
        if bidirectional:
            self.weight_ih_l0_reverse = nn.Parameter(torch.empty(gate_size, input_size))
            self.weight_hh_l0_reverse = nn.Parameter(torch.empty(gate_size, hidden_size))
            self.bias_ih_l0_reverse = nn.Parameter(torch.empty(gate_size))
            self.bias_hh_l0_reverse = nn.Parameter(torch.empty(gate_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same initialization as nn.GRU / GRUCell: uniform(-1/sqrt(H), 1/sqrt(H))
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def _step(self, x: torch.Tensor, h: torch.Tensor, w_ih, w_hh, b_ih, b_hh) -> torch.Tensor:
        # identical math to GRUCell: r, z gates then candidate n, blended hidden
        gi = F.linear(x, w_ih, b_ih)
        gh = F.linear(h, w_hh, b_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        """Run the step-loop on x of shape (B, L, input_size) (or (L, B, ·) when
        batch_first=False).

        h0 follows nn.GRU's contract: (num_layers*num_directions, B, hidden_size).
        With the supported single layer, h0[0] seeds the forward direction and
        h0[1] the backward one; None means zero init (as nn.GRU does).
        """
        # normalize to (B, L, input_size)
        if not self.batch_first:
            x = x.transpose(0, 1)
        B, L, _ = x.shape
        hf = x.new_zeros(B, self.hidden_size) if h0 is None else h0[0]
        f_out = []
        for t in range(L):
            hf = self._step(x[:, t], hf, self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0)
            f_out.append(hf)
        outputs = [torch.stack(f_out, 1)]
        if self.bidirectional:
            hb = x.new_zeros(B, self.hidden_size) if h0 is None else h0[1]
            b_out = []
            for t in range(L - 1, -1, -1):
                hb = self._step(
                    x[:, t],
                    hb,
                    self.weight_ih_l0_reverse,
                    self.weight_hh_l0_reverse,
                    self.bias_ih_l0_reverse,
                    self.bias_hh_l0_reverse,
                )
                b_out.append(hb)
            outputs.append(torch.stack(b_out[::-1], 1))
        out = torch.cat(outputs, dim=-1)
        if not self.batch_first:
            out = out.transpose(0, 1)
        # nn.GRU returns (output, h_n); keep the same contract
        h_n = torch.stack([hf] + ([hb] if self.bidirectional else []), dim=0)
        return out, h_n


class UnfusedLSTM(nn.Module):
    """Unfused full-sequence LSTM: manual step-loop, drop-in for nn.LSTM.

    Supports the subset of nn.LSTM options used by PyPOTS models:
    single layer, optional bidirectional, batch_first.
    Parameter names match nn.LSTM so state_dicts interchange at any nesting depth.
    """

    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        gate_size = 4 * hidden_size
        self.weight_ih_l0 = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(gate_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(gate_size))
        if bidirectional:
            self.weight_ih_l0_reverse = nn.Parameter(torch.empty(gate_size, input_size))
            self.weight_hh_l0_reverse = nn.Parameter(torch.empty(gate_size, hidden_size))
            self.bias_ih_l0_reverse = nn.Parameter(torch.empty(gate_size))
            self.bias_hh_l0_reverse = nn.Parameter(torch.empty(gate_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same initialization as nn.LSTM / LSTMCell: uniform(-1/sqrt(H), 1/sqrt(H))
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def _step(self, x: torch.Tensor, state, w_ih, w_hh, b_ih, b_hh):
        # identical math to LSTMCell: i, f, g, o gates then cell/hidden update
        h, c = state
        gi = F.linear(x, w_ih, b_ih)
        gh = F.linear(h, w_hh, b_hh)
        i, f, g, o = (gi + gh).chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def forward(self, x: torch.Tensor, h0=None):
        """Run the step-loop with nn.LSTM's contract: h0 is (h_0, c_0), each of
        shape (num_layers*num_directions, B, hidden_size); None zero-inits both.
        Returns (output, (h_n, c_n)) like nn.LSTM.

        No in-repo caller yet (no full-sequence nn.LSTM in PyPOTS models);
        kept for API symmetry with gru() and future models.
        """
        if not self.batch_first:
            x = x.transpose(0, 1)
        B, L, _ = x.shape
        cf = x.new_zeros(B, self.hidden_size) if h0 is None else h0[0][1]
        hf = x.new_zeros(B, self.hidden_size) if h0 is None else h0[0][0]
        f_out = []
        for t in range(L):
            hf, cf = self._step(x[:, t], (hf, cf), self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0)
            f_out.append(hf)
        outputs = [torch.stack(f_out, 1)]
        hiddens = [(hf, cf)]
        if self.bidirectional:
            cb = x.new_zeros(B, self.hidden_size) if h0 is None else h0[1][1]
            hb = x.new_zeros(B, self.hidden_size) if h0 is None else h0[1][0]
            b_out = []
            for t in range(L - 1, -1, -1):
                hb, cb = self._step(
                    x[:, t],
                    (hb, cb),
                    self.weight_ih_l0_reverse,
                    self.weight_hh_l0_reverse,
                    self.bias_ih_l0_reverse,
                    self.bias_hh_l0_reverse,
                )
                b_out.append(hb)
            outputs.append(torch.stack(b_out[::-1], 1))
            hiddens.append((hb, cb))
        out = torch.cat(outputs, dim=-1)
        if not self.batch_first:
            out = out.transpose(0, 1)
        # nn.LSTM returns (output, (h_n, c_n)); keep the same contract
        h_n = torch.stack([h[0] for h in hiddens], dim=0)
        c_n = torch.stack([h[1] for h in hiddens], dim=0)
        return out, (h_n, c_n)


def gru(input_size: int, hidden_size: int, bidirectional: bool = False, batch_first: bool = True) -> nn.Module:
    """Device-adaptive nn.GRU: fused on CUDA/CPU, unfused step-loop on NPU."""
    if _use_unfused_rnn():
        return UnfusedGRU(input_size, hidden_size, bidirectional=bidirectional, batch_first=batch_first)
    return nn.GRU(input_size, hidden_size, bidirectional=bidirectional, batch_first=batch_first)


def lstm(input_size: int, hidden_size: int, bidirectional: bool = False, batch_first: bool = True) -> nn.Module:
    """Device-adaptive nn.LSTM: fused on CUDA/CPU, unfused step-loop on NPU."""
    if _use_unfused_rnn():
        return UnfusedLSTM(input_size, hidden_size, bidirectional=bidirectional, batch_first=batch_first)
    return nn.LSTM(input_size, hidden_size, bidirectional=bidirectional, batch_first=batch_first)
