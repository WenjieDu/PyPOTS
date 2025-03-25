"""
The core wrapper assembles the submodules of GP-VAE imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from ...nn.modules import ModelCore
from ...nn.modules.gpvae import BackboneGPVAE


class _GPVAE(ModelCore):
    """model GPVAE with Gaussian Process prior

    Parameters
    ----------
    input_dim : int,
        the feature dimension of the input

    time_length : int,
        the length of each time series

    latent_dim : int,
        the feature dimension of the latent embedding

    encoder_sizes : tuple,
        the tuple of the network size in encoder

    decoder_sizes : tuple,
        the tuple of the network size in decoder

    beta : float,
        the weight of the KL divergence

    M : int,
        the number of Monte Carlo samples for ELBO estimation

    K : int,
        the number of importance weights for IWAE model

    kernel : str,
        the Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        the scale parameter for a kernel function

    length_scale : float,
        the length scale parameter for a kernel function

    kernel_scales : int,
        the number of different length scales over latent space dimensions
    """

    def __init__(
        self,
        input_dim,
        time_length,
        latent_dim,
        encoder_sizes=(64, 64),
        decoder_sizes=(64, 64),
        beta=1,
        M=1,
        K=1,
        kernel="cauchy",
        sigma=1.0,
        length_scale=7.0,
        kernel_scales=1,
        window_size=24,
    ):
        super().__init__()

        self.backbone = BackboneGPVAE(
            input_dim,
            time_length,
            latent_dim,
            encoder_sizes,
            decoder_sizes,
            beta,
            M,
            K,
            kernel,
            sigma,
            length_scale,
            kernel_scales,
            window_size,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
        n_sampling_times=1,
    ):
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        results = {}

        if calc_criterion:
            elbo_loss = self.backbone(X, missing_mask)
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = elbo_loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = elbo_loss
        else:
            imputed_data, imputation_latent = self.backbone.impute(X, missing_mask, n_sampling_times)
            results["imputation"] = imputed_data

            # `imputation_latent` is not "reconstruction" seriously speaking,
            # we just take it to get align with other models' output
            results["reconstruction"] = imputation_latent

        return results
