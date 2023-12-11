"""
The implementation of GRU-D for the partially-observed time-series imputation task.

Refer to the paper "Che, Z., Purushotham, S., Cho, K., Sontag, D.A., & Liu, Y. (2018).
Recurrent Neural Networks for Multivariate Time Series with Missing Values. Scientific Reports."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....nn.modules.rnn import TemporalDecay


class _GRUD(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        device: Union[str, torch.device],
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.device = device

        # create models
        self.rnn_cell = nn.GRUCell(
            self.n_features * 2 + self.rnn_hidden_size, self.rnn_hidden_size
        )
        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True
        )
        self.classifier = nn.Linear(self.rnn_hidden_size, self.n_classes)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        """Forward processing of GRU-D.

        Parameters
        ----------
        inputs :
            The input data.

        training :
            Whether in training mode.

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        values = inputs["X"]
        masks = inputs["missing_mask"]
        deltas = inputs["deltas"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_filledLOCF"]

        hidden_state = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=values.device
        )

        for t in range(self.n_steps):
            # for data, [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap
            x_filledLOCF = X_filledLOCF[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            hidden_state = hidden_state * gamma_h

            x_h = gamma_x * x_filledLOCF + (1 - gamma_x) * empirical_mean
            x_replaced = m * x + (1 - m) * x_h
            data_input = torch.cat([x_replaced, hidden_state, m], dim=1)
            hidden_state = self.rnn_cell(data_input, hidden_state)

        logits = self.classifier(hidden_state)
        classification_pred = torch.softmax(logits, dim=1)
        results = {"classification_pred": classification_pred}

        # if in training mode, return results with losses
        if training:
            torch.log(classification_pred)
            classification_loss = F.nll_loss(
                torch.log(classification_pred), inputs["label"]
            )
            results["loss"] = classification_loss

        return results
