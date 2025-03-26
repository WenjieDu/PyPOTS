"""

"""

# Created by Linglong Qian, Joseph Arul Raj <linglong.qian@kcl.ac.uk, joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

from ...nn.modules import ModelCore
from ...nn.modules.csai.backbone import BackboneBCSAI
from ...nn.modules.loss import Criterion


class _BCSAI(ModelCore):
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
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.step_channels = step_channels
        self.consistency_weight = consistency_weight
        self.imputation_weight = imputation_weight

        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.model = BackboneBCSAI(
            n_steps,
            n_features,
            rnn_hidden_size,
            step_channels,
            training_loss,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
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
            "imputation": imputed_data,
            "consistency_loss": consistency_loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction": (f_reconstruction + b_reconstruction) / 2,
            "f_reconstruction": f_reconstruction,
            "b_reconstruction": b_reconstruction,
        }

        if calc_criterion:
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                loss = consistency_loss + reconstruction_loss
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
                reconstruction = (results["f_reconstruction"] + results["b_reconstruction"]) / 2
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
