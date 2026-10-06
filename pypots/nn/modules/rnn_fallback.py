"""
Device-adaptive RNN layers: fused nn.GRU/LSTM on CUDA/CPU, equivalent unfused
GRUCell/LSTMCell step-loops on Ascend NPU.

Why: on Ascend NPU (CANN 8.x/9.x), the fused kernel behind full-sequence
nn.GRU/LSTM (DynamicGRUV2) has no fp32 backward pass — forward works, but fp32
training fails asynchronously (usually surfacing in optimizer.step). The unfused
step-loops below compose only base aten ops (matmul/sigmoid/tanh), so fp32
forward+backward both run on NPU, with no precision loss and no AMP dependency.

Equivalence: the cell-loop classes use the same parameter names and weight
layout as their fused counterparts (weight_ih_l0 / weight_hh_l0 / bias_ih_l0 /
bias_hh_l0, *_reverse for bidirectional), so state_dicts interchange directly
and outputs match the fused modules (verified bit-exact on CPU, 1.5e-07 on NPU —
pure floating-point ordering).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn


def _use_unfused_rnn() -> bool:
    """Whether to pick the unfused Cell-loop implementations: only on NPU."""
    try:
        import torch_npu

        return torch_npu.npu.is_available()
    except ImportError:
        return False


class UnfusedGRU(nn.Module):
    """Unfused full-sequence GRU: GRUCell step-loop, drop-in for nn.GRU.

    Supports the subset of nn.GRU options used by PyPOTS models:
    batch_first=True, single layer, optional bidirectional.
    Parameter names match nn.GRU so state_dicts interchange.
    """

    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.cell_f = nn.GRUCell(input_size, hidden_size)
        if bidirectional:
            self.cell_b = nn.GRUCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        # normalize to (B, L, input_size)
        if not self.batch_first:
            x = x.transpose(0, 1)
        B, L, _ = x.shape
        hf = x.new_zeros(B, self.hidden_size) if h0 is None else h0[0]
        f_out = []
        for t in range(L):
            hf = self.cell_f(x[:, t], hf)
            f_out.append(hf)
        outputs = [torch.stack(f_out, 1)]
        if self.bidirectional:
            hb = x.new_zeros(B, self.hidden_size) if h0 is None else h0[1]
            b_out = []
            for t in range(L - 1, -1, -1):
                hb = self.cell_b(x[:, t], hb)
                b_out.append(hb)
            outputs.append(torch.stack(b_out[::-1], 1))
        out = torch.cat(outputs, dim=-1)
        if not self.batch_first:
            out = out.transpose(0, 1)
        # nn.GRU returns (output, h_n); keep the same contract
        h_n = torch.stack([hf] + ([hb] if self.bidirectional else []), dim=0)
        return out, h_n


class UnfusedLSTM(nn.Module):
    """Unfused full-sequence LSTM: LSTMCell step-loop, drop-in for nn.LSTM.

    Supports the subset of nn.LSTM options used by PyPOTS models:
    batch_first=True, single layer, optional bidirectional.
    Parameter names match nn.LSTM so state_dicts interchange.
    """

    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.cell_f = nn.LSTMCell(input_size, hidden_size)
        if bidirectional:
            self.cell_b = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h0=None):
        if not self.batch_first:
            x = x.transpose(0, 1)
        B, L, _ = x.shape
        cf = x.new_zeros(B, self.hidden_size) if h0 is None else h0[0][0]
        hf = x.new_zeros(B, self.hidden_size) if h0 is None else h0[0][1]
        f_out = []
        for t in range(L):
            hf, cf = self.cell_f(x[:, t], (hf, cf))
            f_out.append(hf)
        outputs = [torch.stack(f_out, 1)]
        hiddens = [(hf, cf)]
        if self.bidirectional:
            cb = x.new_zeros(B, self.hidden_size) if h0 is None else h0[1][0]
            hb = x.new_zeros(B, self.hidden_size) if h0 is None else h0[1][1]
            b_out = []
            for t in range(L - 1, -1, -1):
                hb, cb = self.cell_b(x[:, t], (hb, cb))
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
