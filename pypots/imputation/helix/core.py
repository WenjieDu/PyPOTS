"""
The core wrapper assembles the submodules of HELIX imputation model
and takes over the forward progress of the algorithm.

Modified to save attention weights for visualization analysis.
"""

# Created by Fengming Zhang <milaogou@gmail.com>
# License: BSD-3-Clause


from ...nn.modules import ModelCore
from ...nn.modules.helix import BackboneHELIX
from ...nn.modules.loss import Criterion


class _HELIX(ModelCore):
    """Core model wrapper for HELIX.

    Modified to provide access to attention weights.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        d_pe: int,
        d_feature_embed: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.training_loss = training_loss

        if validation_metric.__class__.__name__ == "Criterion":
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.backbone = BackboneHELIX(
            n_features=n_features,
            d_pe=d_pe,
            d_feature_embed=d_feature_embed,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    def get_attention_weights(self):
        """
        Get all attention weights from the backbone.

        Returns
        -------
        attention_dict : dict
            Dictionary containing attention weights for each layer and dimension.
        """
        return self.backbone.get_attention_weights()

    def forward(self, inputs: dict, calc_criterion: bool = False) -> dict:
        """
        Parameters
        ----------
        inputs : dict
            Input dictionary containing X, missing_mask, and optionally X_ori, indicating_mask
        calc_criterion : bool
            Whether to calculate loss/metric

        Returns
        -------
        results : dict
            Dictionary containing imputation and optionally loss/metric
        """
        X = inputs["X"]
        missing_mask = inputs["missing_mask"]

        # Forward pass
        reconstruction = self.backbone(X, missing_mask)

        # Replace observed values with original data
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori = inputs["X_ori"]
            indicating_mask = inputs["indicating_mask"]

            if self.training:
                # ORT loss: reconstruction on observed data
                ORT_loss = self.ORT_weight * self.training_loss(reconstruction, X, missing_mask)

                # MIT loss: imputation on artificially masked data
                MIT_loss = self.MIT_weight * self.training_loss(reconstruction, X_ori, indicating_mask)

                loss = ORT_loss + MIT_loss

                results["ORT_loss"] = ORT_loss
                results["MIT_loss"] = MIT_loss
                results["loss"] = loss
            else:
                # Validation metric
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
