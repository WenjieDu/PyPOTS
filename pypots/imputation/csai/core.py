"""

"""

# Created by Linglong Qian, Joseph Arul Raj <linglong.qian@kcl.ac.uk, joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

import torch.nn as nn
from ...nn.modules.csai.backbone import BackboneBCSAI


class _BCSAI(nn.Module):
    """
    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the GRU cell

    step_channels :
        number of channels for each step in the sequence

    consistency_weight :
        weight assigned to the consistency loss during training

    imputation_weight :
        weight assigned to the reconstruction loss during training

    model :
        the underlying BackboneBCSAI model that handles forward and backward pass imputation

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the GRU cell

    step_channels :
        number of channels for each step in the sequence

    consistency_weight :
        weight assigned to the consistency loss

    imputation_weight :
        weight assigned to the reconstruction loss

    Notes
    -----
    CSAI is a bidirectional imputation model that uses forward and backward GRU cells to handle time-series data.
    It computes consistency and reconstruction losses to improve imputation accuracy.
    During training, the forward and backward reconstructions are combined, and losses are used to update the model.
    In evaluation mode, the model also outputs original data and indicating masks for further analysis.

    """

    def __init__(
        self,
        n_steps,
        n_features,
        rnn_hidden_size,
        step_channels,
        consistency_weight,
        imputation_weight,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.step_channels = step_channels
        self.consistency_weight = consistency_weight
        self.imputation_weight = imputation_weight

        self.model = BackboneBCSAI(n_steps, n_features, rnn_hidden_size, step_channels)

    def forward(self, inputs: dict) -> dict:
        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        ) = self.model(inputs)

        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if self.training:
            results["consistency_loss"] = consistency_loss
            results["reconstruction_loss"] = reconstruction_loss
            loss = self.consistency_weight * consistency_loss + self.imputation_weight * reconstruction_loss

            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss
            # results["reconstruction"] = (f_reconstruction + b_reconstruction) / 2
            results["f_reconstruction"] = f_reconstruction
            results["b_reconstruction"] = b_reconstruction
        else:
            results["X_ori"] = inputs["X_ori"]
            results["indicating_mask"] = inputs["indicating_mask"]

        return results
