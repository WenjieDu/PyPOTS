"""
The core wrapper assembles the submodules of TS2Vec vectorizer model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np

from ...nn.modules import ModelCore
from ...nn.modules.ts2vec import TS2VecEncoder
from ...nn.modules.ts2vec.losses import hierarchical_contrastive_loss
from ...nn.modules.ts2vec.utils import take_per_row


class _TS2Vec(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_features: int,
        d_hidden: int,
        n_layers: int,
        mask_mode: str,
        temporal_unit: int = 0,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_features = n_pred_features
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.mask_mode = mask_mode
        self.temporal_unit = temporal_unit

        self.encoder = TS2VecEncoder(
            n_features,
            n_pred_features,
            d_hidden,
            n_layers,
            mask_mode,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
        mask: str = None,
        encoding_window=None,
        causal=False,
        sliding_length=None,
        sliding_padding=0,
    ) -> dict:
        X = inputs["X"]

        results = {}
        reprs = self.encoder.encode(
            X,
            mask,
            encoding_window,
            causal,
            sliding_length,
            sliding_padding,
        )
        results["representation"] = reprs

        if calc_criterion:
            # if in training mode, return results with losses
            n_steps = X.size(1)
            crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=n_steps + 1)
            crop_left = np.random.randint(n_steps - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=n_steps + 1)
            crop_offset = np.random.randint(low=-crop_eleft, high=n_steps - crop_eright + 1, size=X.size(0))

            out1 = self.encoder(take_per_row(X, crop_offset + crop_eleft, crop_right - crop_eleft))
            out1 = out1[:, -crop_l:]
            out2 = self.encoder(take_per_row(X, crop_offset + crop_left, crop_eright - crop_left))
            out2 = out2[:, :crop_l]

            loss = hierarchical_contrastive_loss(out1, out2, temporal_unit=self.temporal_unit)

            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = loss

        return results
