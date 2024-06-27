"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .layers import FourierFilter, MLP, TimeVarKP, TimeInvKP


class BackboneKoopa(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_seg_steps: int,
        d_dynamic: int,
        d_hidden: int,
        n_hidden_layers: int,
        n_blocks: int,
        multistep: bool,
        alpha: int = 0.2,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.alpha = alpha
        self.mask_spectrum = None
        self.disentanglement = FourierFilter(self.mask_spectrum)

        # shared encoder/decoder to make koopman embedding consistent
        self.time_inv_encoder = MLP(
            d_in=n_steps,
            d_out=d_dynamic,
            activation="relu",
            d_hidden=d_hidden,
            n_hidden_layers=n_hidden_layers,
        )
        self.time_inv_decoder = MLP(
            d_in=d_dynamic,
            d_out=n_pred_steps,
            activation="relu",
            d_hidden=d_hidden,
            n_hidden_layers=n_hidden_layers,
        )
        self.time_inv_kps = nn.ModuleList(
            [
                TimeInvKP(
                    input_len=n_steps,
                    pred_len=n_pred_steps,
                    dynamic_dim=d_dynamic,
                    encoder=self.time_inv_encoder,
                    decoder=self.time_inv_decoder,
                )
                for _ in range(n_blocks)
            ]
        )

        # shared encoder/decoder to make koopman embedding consistent
        self.time_var_encoder = MLP(
            d_in=n_seg_steps * n_features,
            d_out=d_dynamic,
            activation="tanh",
            d_hidden=d_hidden,
            n_hidden_layers=n_hidden_layers,
        )
        self.time_var_decoder = MLP(
            d_in=d_dynamic,
            d_out=n_seg_steps * n_features,
            activation="tanh",
            d_hidden=d_hidden,
            n_hidden_layers=n_hidden_layers,
        )
        self.time_var_kps = nn.ModuleList(
            [
                TimeVarKP(
                    enc_in=n_features,
                    input_len=n_steps,
                    pred_len=n_pred_steps,
                    seg_len=n_seg_steps,
                    dynamic_dim=d_dynamic,
                    encoder=self.time_var_encoder,
                    decoder=self.time_var_decoder,
                    multistep=multistep,
                )
                for _ in range(n_blocks)
            ]
        )

    def _get_mask_spectrum(self, train_dataloader):
        """
        get shared frequency spectrums
        """
        amps = 0.0
        for _, data in enumerate(train_dataloader):
            lookback_window = data[1]  # X is at index 1
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)
        mask_spectrum = amps.topk(int(amps.shape[0] * self.alpha)).indices
        return mask_spectrum  # as the spectrums of time-invariant component

    def init_mask_spectrum(self, train_dataloader: DataLoader):
        self.mask_spectrum = self._get_mask_spectrum(train_dataloader)

    def forward(self, X):
        assert (
            self.mask_spectrum is not None
        ), "Please initialize the mask spectrum first with init_mask_spectrum() method."

        residual, output = X, None
        for i in range(self.n_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if output is None:
                output = time_inv_output + time_var_output
            else:
                output += time_inv_output + time_var_output

        return output
