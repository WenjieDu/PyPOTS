"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from .layers import SeasonalPrediction
from ..fedformer.layers import SeriesDecompositionMultiBlock


class BackboneMICN(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_pred_steps,
        n_pred_features,
        n_layers,
        d_model,
        conv_kernel=[12, 24],
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features

        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((n_steps + n_pred_steps + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((n_steps + n_pred_steps + ii - 1) // ii)

        self.decomp_multi = SeriesDecompositionMultiBlock(decomp_kernel)

        self.conv_trans = SeasonalPrediction(
            embedding_size=d_model,
            d_layers=n_layers,
            decomp_kernel=decomp_kernel,
            c_out=n_pred_features,
            conv_kernel=conv_kernel,
            isometric_kernel=isometric_kernel,
        )
