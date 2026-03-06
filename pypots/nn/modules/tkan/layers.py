"""
The layers for TKAN (Temporal Kolmogorov-Arnold Networks).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """A KAN (Kolmogorov-Arnold Network) linear layer using B-spline activations.

    This is a PyTorch implementation inspired by the efficient-kan approach
    (https://github.com/Blealtan/efficient-kan), as used in the TKAN paper
    :cite:`genet2024tkan`.

    Parameters
    ----------
    in_features :
        Number of input features.

    out_features :
        Number of output features.

    grid_size :
        The number of grid intervals for the B-spline basis. Default: 3.

    spline_order :
        The order of the B-spline. Default: 3.

    base_activation :
        Name of the base activation function. Default: "relu".

    grid_range :
        The range of the grid for the B-spline basis. Default: (-1.0, 1.0).

    use_layernorm :
        Whether to apply layer normalization on the input before processing. Default: True.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 3,
        spline_order: int = 3,
        base_activation: str = "relu",
        grid_range: tuple = (-1.0, 1.0),
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation_name = base_activation
        self.grid_range = grid_range
        self.use_layernorm = use_layernorm

        # Build B-spline grid (non-trainable)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        # shape: [1, in_features, grid_pts]
        grid = grid.unsqueeze(0).unsqueeze(0).expand(1, in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # Base activation weight: relu(x) -> out_features
        self.base_weight = nn.Parameter(torch.empty(in_features, out_features))
        # Spline weight: b_splines(x) -> out_features
        self.spline_weight = nn.Parameter(torch.empty(in_features * (grid_size + spline_order), out_features))

        if use_layernorm:
            self.layer_norm = nn.LayerNorm(in_features)
        else:
            self.layer_norm = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis values for input x.

        Parameters
        ----------
        x : shape [batch, in_features]

        Returns
        -------
        Tensor of shape [batch, in_features * (grid_size + spline_order)]
        """
        # x: [batch, in_features] -> [batch, in_features, 1]
        x = x.unsqueeze(-1)
        # grid: [1, in_features, grid_pts]
        grid = self.grid

        # order-0 basis: [batch, in_features, grid_pts - 1]
        bases = ((x >= grid[..., :-1]) & (x < grid[..., 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            left_denom = grid[..., k:-1] - grid[..., : -(k + 1)]
            right_denom = grid[..., k + 1 :] - grid[..., 1:-k]

            # Avoid division by zero
            left_denom = left_denom + (left_denom == 0).to(x.dtype) * 1e-6
            right_denom = right_denom + (right_denom == 0).to(x.dtype) * 1e-6

            left = (x - grid[..., : -(k + 1)]) / left_denom
            right = (grid[..., k + 1 :] - x) / right_denom
            bases = left * bases[..., :-1] + right * bases[..., 1:]

        # bases: [batch, in_features, grid_size + spline_order]
        batch_size = x.size(0)
        return bases.reshape(batch_size, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : shape [batch, in_features]

        Returns
        -------
        Tensor of shape [batch, out_features]
        """
        if self.use_layernorm and self.layer_norm is not None:
            x = self.layer_norm(x)

        base_activation_fn = getattr(F, self.base_activation_name)
        base_output = base_activation_fn(x) @ self.base_weight
        spline_output = self.b_splines(x) @ self.spline_weight
        return base_output + spline_output


class TKANCell(nn.Module):
    """A single TKAN (Temporal Kolmogorov-Arnold Networks) cell.

    This cell implements a modified LSTM cell where the output gate is replaced
    by KAN-based sub-layers. Inspired by :cite:`genet2024tkan`.

    Parameters
    ----------
    in_features :
        Number of input features.

    hidden_size :
        Dimensionality of the hidden/output state.

    sub_kan_configs :
        Configurations for the KAN sub-layers. Each element can be:
        - None: default KANLinear with grid_size=3, spline_order=3
        - int/float: KANLinear with that spline_order
        - dict: KANLinear with provided keyword arguments
        - str: a standard activation name using a simple Linear layer (e.g. "relu", "tanh")
        Default: [None] (one sub-layer with default config).

    sub_kan_output_dim :
        Output dimension of each sub-KAN layer. Defaults to ``in_features``.

    sub_kan_input_dim :
        Input dimension of each sub-KAN layer. Defaults to ``in_features``.

    use_bias :
        Whether to use bias in the gate linear layers. Default: True.

    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        sub_kan_configs: Optional[List] = None,
        sub_kan_output_dim: Optional[int] = None,
        sub_kan_input_dim: Optional[int] = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.sub_kan_configs = sub_kan_configs if sub_kan_configs is not None else [None]

        # Sub-KAN dimensions default to in_features
        self.sub_kan_input_dim = sub_kan_input_dim if sub_kan_input_dim is not None else in_features
        self.sub_kan_output_dim = sub_kan_output_dim if sub_kan_output_dim is not None else in_features

        self.use_bias = use_bias
        n_sub = len(self.sub_kan_configs)

        # Gate projection weights (i, f, g — 3 gates like LSTM but all use sigmoid)
        self.kernel = nn.Parameter(torch.empty(in_features, hidden_size * 3))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 3))
        if use_bias:
            bias = torch.zeros(hidden_size * 3)
            # Unit forget bias initialization: set forget gate bias to 1
            bias[hidden_size : hidden_size * 2] = 1.0
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

        # Sub-KAN layers
        self.tkan_sub_layers = nn.ModuleList()
        for config in self.sub_kan_configs:
            layer = self._build_sub_layer(config)
            self.tkan_sub_layers.append(layer)

        # Recurrent mixing kernels for sub-layers:
        # sub_tkan_recurrent_kernel_inputs[i]: [in_features, sub_kan_input_dim]
        # sub_tkan_recurrent_kernel_states[i]: [sub_kan_output_dim, sub_kan_input_dim]
        self.sub_tkan_recurrent_kernel_inputs = nn.Parameter(torch.empty(n_sub, in_features, self.sub_kan_input_dim))
        self.sub_tkan_recurrent_kernel_states = nn.Parameter(
            torch.empty(n_sub, self.sub_kan_output_dim, self.sub_kan_input_dim)
        )

        # Sub-layer output mixing coefficients:
        # sub_tkan_kernel[i]: [sub_kan_output_dim * 2]  (split into h_coef and x_coef)
        self.sub_tkan_kernel = nn.Parameter(torch.empty(n_sub, self.sub_kan_output_dim * 2))

        # Aggregation from all sub-layer outputs to output gate
        self.aggregated_weight = nn.Parameter(torch.empty(n_sub * self.sub_kan_output_dim, hidden_size))
        self.aggregated_bias = nn.Parameter(torch.zeros(hidden_size))

        self._reset_parameters()

    def _build_sub_layer(self, config) -> nn.Module:
        """Build a single sub-KAN layer from a config descriptor."""
        in_dim = self.sub_kan_input_dim
        out_dim = self.sub_kan_output_dim

        if config is None:
            return KANLinear(in_dim, out_dim, grid_size=3, spline_order=3, use_layernorm=True)
        elif isinstance(config, (int, float)):
            spline_order = int(config)
            return KANLinear(in_dim, out_dim, spline_order=spline_order, use_layernorm=True)
        elif isinstance(config, dict):
            # Allow the dict to override use_layernorm; default to True if not specified
            cfg = dict(config)
            cfg.setdefault("use_layernorm", True)
            return KANLinear(in_dim, out_dim, **cfg)
        elif isinstance(config, str):
            # Use a simple nn.Linear + functional activation
            act_fn = getattr(F, config, None)
            if act_fn is None or not callable(act_fn):
                raise ValueError(
                    f"Unknown activation '{config}'. Provide a valid torch.nn.functional name "
                    f"(e.g. 'relu', 'tanh', 'sigmoid', 'gelu')."
                )

            class _LinearActivation(nn.Module):
                def __init__(self, in_d, out_d, activation):
                    super().__init__()
                    self.linear = nn.Linear(in_d, out_d)
                    self.activation = activation

                def forward(self, x):
                    return self.activation(self.linear(x))

            return _LinearActivation(in_dim, out_dim, act_fn)
        else:
            raise ValueError(f"Unsupported sub_kan_config type: {type(config)}")

    def _reset_parameters(self):
        nn.init.orthogonal_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        nn.init.orthogonal_(self.sub_tkan_recurrent_kernel_inputs.view(-1, self.sub_kan_input_dim))
        nn.init.orthogonal_(self.sub_tkan_recurrent_kernel_states.view(-1, self.sub_kan_input_dim))
        nn.init.orthogonal_(self.sub_tkan_kernel)
        nn.init.xavier_uniform_(self.aggregated_weight)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        sub_states: List[torch.Tensor],
    ):
        """One step of the TKAN cell.

        Parameters
        ----------
        x : shape [batch, in_features]
        h : shape [batch, hidden_size]  — previous hidden state
        c : shape [batch, hidden_size]  — previous cell state
        sub_states : list of tensors, each [batch, sub_kan_output_dim]

        Returns
        -------
        h_new : [batch, hidden_size]
        c_new : [batch, hidden_size]
        new_sub_states : list of [batch, sub_kan_output_dim]
        """
        # Gate computation (like LSTM but all three use sigmoid)
        gates = x @ self.kernel + h @ self.recurrent_kernel
        if self.bias is not None:
            gates = gates + self.bias
        gates = torch.sigmoid(gates)
        i, f, g = gates.chunk(3, dim=-1)

        # New cell state
        c_new = f * c + i * torch.tanh(g)

        # Process sub-KAN layers
        sub_outputs = []
        new_sub_states = []
        for idx, (sub_layer, sub_state) in enumerate(zip(self.tkan_sub_layers, sub_states)):
            sub_kernel_x = self.sub_tkan_recurrent_kernel_inputs[idx]  # [in_features, sub_kan_input_dim]
            sub_kernel_h = self.sub_tkan_recurrent_kernel_states[idx]  # [sub_kan_output_dim, sub_kan_input_dim]
            agg_input = x @ sub_kernel_x + sub_state @ sub_kernel_h  # [batch, sub_kan_input_dim]
            sub_output = sub_layer(agg_input)  # [batch, sub_kan_output_dim]

            # Update sub-state
            sub_mix = self.sub_tkan_kernel[idx]  # [sub_kan_output_dim * 2]
            coef_h = sub_mix[: self.sub_kan_output_dim]  # coefficient for sub_output
            coef_x = sub_mix[self.sub_kan_output_dim :]  # coefficient for old sub_state
            new_sub_state = coef_h * sub_output + coef_x * sub_state

            sub_outputs.append(sub_output)
            new_sub_states.append(new_sub_state)

        # Output gate: aggregate all sub-layer outputs
        aggregated = torch.cat(sub_outputs, dim=-1)  # [batch, n_sub * sub_kan_output_dim]
        o = torch.sigmoid(aggregated @ self.aggregated_weight + self.aggregated_bias)

        # New hidden state
        h_new = o * torch.tanh(c_new)

        return h_new, c_new, new_sub_states

    def init_hidden_states(self, batch_size: int, device: torch.device):
        """Initialize hidden states to zeros.

        Returns
        -------
        h : [batch, hidden_size]
        c : [batch, hidden_size]
        sub_states : list of [batch, sub_kan_output_dim]
        """
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        sub_states = [torch.zeros(batch_size, self.sub_kan_output_dim, device=device) for _ in self.tkan_sub_layers]
        return h, c, sub_states
