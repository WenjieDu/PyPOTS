"""Regression test for FEDformerDecoder.forward wiring (issue #970)."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch

from pypots.nn.modules.fedformer import FEDformerDecoder, FEDformerEncoder


def test_decoder_reads_memory_and_trend():
    torch.manual_seed(0)
    kwargs = {"n_heads": 8, "d_model": 16, "d_ffn": 16, "moving_avg_window_size": 3, "dropout": 0.0, "modes": 4}
    encoder = FEDformerEncoder(n_steps=16, n_layers=1, **kwargs).eval()
    decoder = FEDformerDecoder(n_steps=16, n_pred_steps=4, n_layers=1, d_output=3, **kwargs).eval()
    memory, _ = encoder(torch.randn(2, 16, 16))
    x = torch.randn(2, 12, 16)
    with torch.no_grad():
        seasonal, trend = decoder(x, memory)
        assert seasonal.shape == trend.shape == (2, 12, 3)
        # the memory now reaches cross-attention, and an initial trend is carried through
        assert not torch.allclose(decoder(x, memory + torch.randn_like(memory))[0], seasonal)
        shifted_trend = decoder(x, memory, trend=torch.ones(2, 12, 3))[1]
        torch.testing.assert_close(shifted_trend, trend + 1.0)
