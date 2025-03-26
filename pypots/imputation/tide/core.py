"""
The core wrapper assembles the submodules of TiDE imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding
from ...nn.modules.tide.autoencoder import TiDE


class _TiDE(ModelCore):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        d_model,
        d_hidden,
        d_feature_encode,
        d_temporal_decoder_hidden,
        dropout,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=False,
            dropout=dropout,
        )
        # n_output_steps = n_steps
        # n_output_features = n_features
        # d_flatten = n_steps + (n_steps + n_output_steps) * d_feature_encode
        # self.feature_encoder = ResBlock(
        #     n_features,
        #     d_hidden,
        #     d_feature_encode,
        #     dropout,
        # )
        # self.encoder = TideEncoder(
        #     n_steps,
        #     n_features,
        #     n_layers,
        #     d_flatten,
        #     d_hidden,
        #     dropout,
        # )
        # self.decoder = TideDecoder(
        #     n_steps,
        #     n_steps,
        #     n_output_features,
        #     n_layers,
        #     d_hidden,
        #     d_feature_encode,
        #     dropout,
        # )
        # self.temporal_decoder = ResBlock(
        #     n_output_features + d_feature_encode,
        #     d_temporal_decoder_hidden,
        #     1,
        #     dropout,
        # )
        # self.residual_proj = nn.Linear(n_steps, n_output_steps)
        self.tide = TiDE(
            n_steps,
            n_features,
            n_layers,
            d_hidden,
            d_feature_encode,
            d_temporal_decoder_hidden,
            dropout,
        )

        # for the imputation task, the output dim is the same as input dim
        # self.output_projection = nn.Linear(d_model, n_features)

        # apply SAITS loss function to TiDE on the imputation task
        self.training_loss = SaitsLoss(ORT_weight, MIT_weight, training_loss)
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # # WDU: the original TiDE paper isn't proposed for imputation task. Hence the model doesn't take
        # # the missing mask into account, which means, in the process, the model doesn't know which part of
        # # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # # the output layers to project back from the hidden space to the original space.
        # X_enc = self.saits_embedding(X, missing_mask)
        #
        # # TiDE encoder processing
        # feature = self.feature_encoder(missing_mask)
        # # print(f"X.shape: {X.shape}")
        # # print(f"feature.shape: {feature.shape}")
        # enc_in = torch.cat([X_enc, feature], dim=-1)
        # enc_out = self.encoder(enc_in)
        # dec_out = self.decoder(enc_out)
        # reconstruction = self.temporal_decoder(
        #     torch.cat([feature[:, self.n_steps :], dec_out], dim=-1)
        # ).squeeze(-1) + self.residual_proj(X_enc)
        # # # project back the original data space
        # # reconstruction = self.output_projection(enc_out)
        reconstruction = self.tide(X, missing_mask)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                loss, ORT_loss, MIT_loss = self.training_loss(reconstruction, X_ori, missing_mask, indicating_mask)
                results["ORT_loss"] = ORT_loss
                results["MIT_loss"] = MIT_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
