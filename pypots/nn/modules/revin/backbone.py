"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .layers import RevIN
from ..scinet import BackboneSCINet


class BackboneRevINSCINet(BackboneSCINet):
    def __init__(
        self,
        n_out_steps,
        n_in_steps,
        n_in_features,
        d_hidden,
        n_stacks,
        n_levels,
        n_decoder_layers,
        n_groups,
        kernel_size=5,
        dropout: float = 0.5,
        concat_len: int = 0,
        pos_enc: bool = False,
        modified: bool = True,
        single_step_output_One: bool = False,
    ):
        super().__init__(
            n_out_steps,
            n_in_steps,
            n_in_features,
            d_hidden,
            n_stacks,
            n_levels,
            n_decoder_layers,
            n_groups,
            kernel_size,
            dropout,
            concat_len,
            pos_enc,
            modified,
            single_step_output_One,
        )

        self.revin = RevIN(n_in_features)

    def forward(self, x):
        x = self.revin(x, mode="norm")
        x, _ = super().forward(x)
        x = self.revin(x, mode="denorm")
        return x
