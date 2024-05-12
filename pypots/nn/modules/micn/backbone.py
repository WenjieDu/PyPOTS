"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from .layers import SeasonalPrediction


class BackboneMICN(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_pred_steps,
        n_pred_features,
        n_layers,
        d_model,
        decomp_kernel,
        isometric_kernel,
        conv_kernel: list,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features

        self.conv_trans = SeasonalPrediction(
            embedding_size=d_model,
            d_layers=n_layers,
            decomp_kernel=decomp_kernel,
            c_out=n_pred_features,
            conv_kernel=conv_kernel,
            isometric_kernel=isometric_kernel,
        )

    def forward(self, x):
        dec_out = self.conv_trans(x)
        return dec_out
