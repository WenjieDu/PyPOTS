"""
The core wrapper assembles the submodules of GRU-D imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch.nn as nn

from ...nn.functional import calc_mse
from ...nn.modules.grud import BackboneGRUD


class _GRUD(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        # create models
        self.backbone = BackboneGRUD(
            n_steps,
            n_features,
            rnn_hidden_size,
        )
        self.output_projection = nn.Linear(rnn_hidden_size, n_features)

    def forward(self, inputs: dict) -> dict:
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
        X = inputs["X"]
        missing_mask = inputs["missing_mask"]
        deltas = inputs["deltas"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_filledLOCF"]

        hidden_states, _ = self.backbone(X, missing_mask, deltas, empirical_mean, X_filledLOCF)

        # project back the original data space
        reconstruction = self.output_projection(hidden_states)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if self.training:
            results["loss"] = calc_mse(reconstruction, X, missing_mask)

        return results
