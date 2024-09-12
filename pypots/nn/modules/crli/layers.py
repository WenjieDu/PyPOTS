"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple

import torch
import torch.nn as nn

RNN_CELL = {
    "LSTM": nn.LSTMCell,
    "GRU": nn.GRUCell,
}


def reverse_tensor(tensor_: torch.Tensor) -> torch.Tensor:
    if tensor_.dim() <= 1:
        return tensor_
    indices = range(tensor_.size()[1])[::-1]
    indices = torch.tensor(indices, dtype=torch.long, device=tensor_.device, requires_grad=False)
    return tensor_.index_select(1, indices)


class MultiRNNCell(nn.Module):
    def __init__(
        self,
        cell_type: str,
        n_layer: int,
        d_input: int,
        d_hidden: int,
    ):
        super().__init__()
        self.cell_type = cell_type
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_hidden = d_hidden

        self.model = nn.ModuleList()
        if cell_type in ["LSTM", "GRU"]:
            for i in range(n_layer):
                if i == 0:
                    self.model.append(RNN_CELL[cell_type](d_input, d_hidden))
                else:
                    self.model.append(RNN_CELL[cell_type](d_hidden, d_hidden))

        self.output_layer = nn.Linear(d_hidden, d_input)

    def forward(self, X: torch.Tensor, missing_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        bz, n_steps, _ = X.shape
        device = X.device

        hidden_state = torch.zeros((bz, self.d_hidden), device=device)
        hidden_state_collector = torch.empty((bz, n_steps, self.d_hidden), device=device)
        output_collector = torch.empty((bz, n_steps, self.d_input), device=device)
        if self.cell_type == "LSTM":
            cell_states = [torch.zeros((bz, self.d_hidden), device=device) for _ in range(self.n_layer)]

            for step in range(n_steps):
                x = X[:, step, :]
                estimation = self.output_layer(hidden_state)
                output_collector[:, step] = estimation
                imputed_x = missing_mask[:, step] * x + (1 - missing_mask[:, step]) * estimation
                for i in range(self.n_layer):
                    if i == 0:
                        hidden_state, cell_state = self.model[i](imputed_x, (hidden_state, cell_states[i]))
                    else:
                        hidden_state, cell_state = self.model[i](hidden_state, (hidden_state, cell_states[i]))

                hidden_state_collector[:, step, :] = hidden_state

        elif self.cell_type == "GRU":
            for step in range(n_steps):
                x = X[:, step, :]
                estimation = self.output_layer(hidden_state)
                output_collector[:, step] = estimation
                imputed_x = missing_mask[:, step] * x + (1 - missing_mask[:, step]) * estimation
                for i in range(self.n_layer):
                    if i == 0:
                        hidden_state = self.model[i](imputed_x, hidden_state)
                    else:
                        hidden_state = self.model[i](hidden_state, hidden_state)

                hidden_state_collector[:, step, :] = hidden_state

        output_collector = output_collector[:, 1:]
        estimation = self.output_layer(hidden_state).unsqueeze(1)
        output_collector = torch.concat([output_collector, estimation], dim=1)
        return output_collector, hidden_state


class CrliGenerator(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_features: int,
        d_hidden: int,
        cell_type: str,
    ):
        super().__init__()
        self.f_rnn = MultiRNNCell(cell_type, n_layers, n_features, d_hidden)
        self.b_rnn = MultiRNNCell(cell_type, n_layers, n_features, d_hidden)

    def forward(self, X: torch.Tensor, missing_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_outputs, f_final_hidden_state = self.f_rnn(X, missing_mask)
        b_outputs, b_final_hidden_state = self.b_rnn(X, missing_mask)
        b_outputs = reverse_tensor(b_outputs)  # reverse the output of the backward rnn
        imputation_latent = (f_outputs + b_outputs) / 2
        fb_final_hidden_states = torch.concat([f_final_hidden_state, b_final_hidden_state], dim=-1)
        return imputation_latent, fb_final_hidden_states


class CrliDiscriminator(nn.Module):
    def __init__(
        self,
        cell_type: str,
        d_input: int,
    ):
        super().__init__()
        self.cell_type = cell_type
        # this setting is the same with the official implementation
        self.rnn_cell_module_list = nn.ModuleList(
            [
                RNN_CELL[cell_type](d_input, 32),
                RNN_CELL[cell_type](32, 16),
                RNN_CELL[cell_type](16, 8),
                RNN_CELL[cell_type](8, 16),
                RNN_CELL[cell_type](16, 32),
            ]
        )
        self.output_layer = nn.Linear(32, d_input)

    def forward(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
        imputation_latent: torch.Tensor,
    ) -> torch.Tensor:
        imputed_X = (X * missing_mask) + (imputation_latent * (1 - missing_mask))

        bz, n_steps, _ = imputed_X.shape
        device = imputed_X.device
        hidden_states = [
            torch.zeros((bz, 32), device=device),
            torch.zeros((bz, 16), device=device),
            torch.zeros((bz, 8), device=device),
            torch.zeros((bz, 16), device=device),
            torch.zeros((bz, 32), device=device),
        ]
        hidden_state_collector = torch.empty((bz, n_steps, 32), device=device)
        if self.cell_type == "LSTM":
            cell_states = [
                torch.zeros((bz, 32), device=device),
                torch.zeros((bz, 16), device=device),
                torch.zeros((bz, 8), device=device),
                torch.zeros((bz, 16), device=device),
                torch.zeros((bz, 32), device=device),
            ]
            for step in range(n_steps):
                x = imputed_X[:, step, :]
                for i, rnn_cell in enumerate(self.rnn_cell_module_list):
                    if i == 0:
                        hidden_state, cell_state = rnn_cell(x, (hidden_states[i], cell_states[i]))
                    else:
                        hidden_state, cell_state = rnn_cell(hidden_states[i - 1], (hidden_states[i], cell_states[i]))
                    cell_states[i] = cell_state
                    hidden_states[i] = hidden_state

                hidden_state_collector[:, step, :] = hidden_state

        elif self.cell_type == "GRU":
            for step in range(n_steps):
                x = imputed_X[:, step, :]
                for i, rnn_cell in enumerate(self.rnn_cell_module_list):
                    if i == 0:
                        hidden_state = rnn_cell(x, hidden_states[i])
                    else:
                        hidden_state = rnn_cell(hidden_states[i - 1], hidden_states[i])
                    hidden_states[i] = hidden_state
                hidden_state_collector[:, step, :] = hidden_state

        output_collector = self.output_layer(hidden_state_collector)
        return output_collector


class CrliDecoder(nn.Module):
    def __init__(
        self,
        n_steps: int,
        d_input: int,
        d_output: int,
        fcn_output_dims: list = None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.d_output = d_output

        if fcn_output_dims is None:
            fcn_output_dims = [d_input]
        self.fcn_output_dims = fcn_output_dims

        self.fcn = nn.ModuleList()
        for output_dim in fcn_output_dims:
            self.fcn.append(nn.Linear(d_input, output_dim))
            d_input = output_dim

        self.rnn_cell = nn.GRUCell(fcn_output_dims[-1], fcn_output_dims[-1])
        self.output_layer = nn.Linear(fcn_output_dims[-1], d_output)

    def forward(self, generator_fb_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = generator_fb_hidden_states.device
        bz, _ = generator_fb_hidden_states.shape

        fcn_latent = generator_fb_hidden_states
        for layer in self.fcn:
            fcn_latent = layer(fcn_latent)
        hidden_state = fcn_latent
        hidden_state_collector = torch.empty((bz, self.n_steps, self.fcn_output_dims[-1]), device=device)
        for i in range(self.n_steps):
            hidden_state = self.rnn_cell(hidden_state, hidden_state)
            hidden_state_collector[:, i, :] = hidden_state
        reconstruction = self.output_layer(hidden_state_collector)
        return reconstruction, fcn_latent
