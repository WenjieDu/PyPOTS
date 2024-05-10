"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .layers import RevIN
from ..scinet import BackboneSCINet


class BackboneRevINSCINet(BackboneSCINet):
    def __init__(
        self,
        output_len,
        input_len,
        input_dim=9,
        hid_size=1,
        num_stacks=1,
        num_levels=3,
        num_decoder_layer=1,
        concat_len=0,
        groups=1,
        kernel=5,
        dropout=0.5,
        single_step_output_One=0,
        positionalE=False,
        modified=True,
    ):
        super().__init__(
            output_len,
            input_len,
            input_dim,
            hid_size,
            num_stacks,
            num_levels,
            num_decoder_layer,
            concat_len,
            groups,
            kernel,
            dropout,
            single_step_output_One,
            positionalE,
            modified,
        )

        self.revin = RevIN(input_dim)

    def forward(self, x):
        x = self.revin(x, mode="norm")
        x, _ = super().forward(x)
        x = self.revin(x, mode="denorm")
        return x
