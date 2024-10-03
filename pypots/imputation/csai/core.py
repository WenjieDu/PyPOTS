"""

"""

# Created by Linglong Qian, Joseph Arul Raj <linglong.qian@kcl.ac.uk, joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

import torch.nn as nn
from ...nn.modules.csai.backbone import BackboneBCSAI


class _BCSAI(nn.Module):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    """
    def __init__(self, n_steps, 
                 n_features, 
                 rnn_hidden_size, 
                 step_channels, 
                 intervals, 
                 consistency_weight, 
                 imputation_weight,
                 device) :
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.step_channels = step_channels
        self.intervals = intervals
        self.consistency_weight = consistency_weight
        self.imputation_weight = imputation_weight
        self.device = device
        
        self.model = BackboneBCSAI(n_steps, n_features, rnn_hidden_size, step_channels, intervals, self.device)

    def forward(self, inputs:dict, training:bool = True) -> dict:
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
        if training:
            results["consistency_loss"] = consistency_loss
            results["reconstruction_loss"] = reconstruction_loss
            loss = self.consistency_weight * consistency_loss + self.imputation_weight * reconstruction_loss

            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss
            # results["reconstruction"] = (f_reconstruction + b_reconstruction) / 2
            results["f_reconstruction"] = f_reconstruction
            results["b_reconstruction"] = b_reconstruction
        if not training:
            results["X_ori"] = inputs["X_ori"]
            results["indicating_mask"] = inputs["indicating_mask"]

        return results
