"""Regression test for residual connections in the Transformer decoder (issue #967)."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch

from pypots.nn.modules.transformer import TransformerDecoder


def test_decoder_output_depends_on_its_input_with_constant_memory():
    # Without residual connections the decoder input reached the output only via attention weights, so a constant
    # memory made the output independent of it.
    torch.manual_seed(0)
    decoder = TransformerDecoder(1, 8, 2, 4, 4, 16, 0.0, 0.0).eval()
    x, memory = torch.randn(2, 4, 8), torch.ones(2, 5, 8)
    with torch.no_grad():
        assert not torch.allclose(decoder(x, memory)[0], decoder(x + 1.0, memory)[0])
