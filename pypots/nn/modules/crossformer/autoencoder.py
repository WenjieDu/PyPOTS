"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn
from einops import rearrange


class CrossformerEncoder(nn.Module):
    def __init__(self, attn_layers):
        super().__init__()
        self.encode_blocks = nn.ModuleList(attn_layers)

    def forward(self, x, src_mask=None):
        attn_weights_collector = []
        enc_output = x

        for block in self.encode_blocks:
            enc_output, attn_weights = block(enc_output, src_mask)
            attn_weights_collector.append(attn_weights)

        return enc_output, attn_weights_collector


class CrossformerDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.decode_layers = nn.ModuleList(layers)

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(
            final_predict,
            "b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d",
            out_d=ts_d,
        )

        return final_predict
