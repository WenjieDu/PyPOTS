"""Regression tests for FEDformer head counts (issue #971)."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import torch

from pypots.imputation import FEDformer
from pypots.nn.modules.fedformer import FEDformerDecoder, FEDformerEncoder
from pypots.nn.modules.fedformer.layers import FourierBlock


def test_default_weights_keep_their_shape():
    # Checkpoints trained with n_heads=8 must still load.
    assert tuple(FourierBlock(16, 16, 16, modes=4).weights1.shape) == (8, 2, 2, 4)
    with pytest.raises(ValueError, match="divisible by n_heads"):
        FourierBlock(16, 16, 16, modes=4, n_heads=3)


@pytest.mark.parametrize("n_heads", [2, 4, 16])
@pytest.mark.parametrize("version", ["Fourier", "Wavelets"])
def test_encoder_and_decoder_accept_other_head_counts(n_heads, version):
    kwargs = {"n_heads": n_heads, "d_model": 16, "d_ffn": 16, "moving_avg_window_size": 3, "dropout": 0.0}
    kwargs.update(version=version, modes=4)
    encoder = FEDformerEncoder(n_steps=16, n_layers=1, **kwargs)
    decoder = FEDformerDecoder(n_steps=16, n_pred_steps=4, n_layers=1, d_output=3, **kwargs)
    memory, _ = encoder(torch.randn(2, 16, 16))
    seasonal, trend = decoder(torch.randn(2, 12, 16), memory)
    assert memory.shape == (2, 16, 16) and seasonal.shape == trend.shape == (2, 12, 3)


def test_fedformer_imputation_trains_with_four_heads():
    X = np.random.default_rng(0).normal(size=(8, 16, 4))
    model = FEDformer(
        n_steps=16,
        n_features=4,
        n_layers=1,
        n_heads=4,
        d_model=16,
        d_ffn=16,
        moving_avg_window_size=3,
        dropout=0,
        modes=4,
        epochs=1,
        device="cpu",
        verbose=False,
    )
    model.fit({"X": X})
    assert np.isfinite(model.predict({"X": X})["imputation"]).all()
