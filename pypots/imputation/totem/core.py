"""
The core wrapper assembles the submodules of TOTEM imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.functional.error import calc_mse
from ...nn.modules import ModelCore
from ...nn.modules.totem import VQVAE


class _TOTEM(ModelCore):
    def __init__(
        self,
        n_steps,
        n_features,
        block_hidden_size,
        num_residual_layers,
        res_hidden_size,
        embedding_dim,
        num_embeddings,
        commitment_cost,
        compression_factor,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features

        self.autoencoder = VQVAE(
            block_hidden_size,
            num_residual_layers,
            res_hidden_size,
            embedding_dim,
            num_embeddings,
            commitment_cost,
            compression_factor,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # TOTEM encoder processing
        flatten_X = X.permute(0, 2, 1).reshape(-1, self.n_steps)
        reconstruction, vq_loss, perplexity, embedding_weight, encoding_indices, encodings = self.autoencoder(flatten_X)
        reconstruction = reconstruction.reshape(-1, self.n_features, self.n_steps).permute(0, 2, 1)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori = inputs["X_ori"]
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                recon_loss = calc_mse(reconstruction, X_ori)
                results["vq_loss"] = vq_loss
                results["recon_loss"] = recon_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = recon_loss + vq_loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = calc_mse(reconstruction, X_ori) + vq_loss

        return results
