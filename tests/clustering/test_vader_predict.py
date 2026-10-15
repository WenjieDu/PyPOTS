"""Regression tests for VaDER inference with singleton batches, without training."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import torch

from pypots.clustering import VaDER


class _GaussianCore(torch.nn.Module):
    def forward(self, inputs):
        X = inputs["X"]
        mu_tilde = torch.stack((X[:, 0, 0], torch.zeros_like(X[:, 0, 0])), dim=1)
        return {
            "mu_tilde": mu_tilde,
            "stddev_tilde": torch.ones_like(mu_tilde),
            "mu": X.new_tensor([[-1.0, 0.0], [1.0, 0.0]]),
            "var": X.new_ones((2, 2)),
            "phi": X.new_full((2,), 0.5),
            "z": mu_tilde,
            "imputation_latent": X,
        }


@pytest.mark.xdist_group(name="clustering-vader")
@pytest.mark.parametrize("n_samples,batch_size", [(1, 8), (8, 8), (9, 8), (17, 8), (9, 1)])
@pytest.mark.parametrize("return_latent_vars", [False, True])
def test_vader_predict_singleton_batches(n_samples, batch_size, return_latent_vars, monkeypatch):
    def forbid_fit(*args, **kwargs):
        pytest.fail("Prediction must not train the model")

    monkeypatch.setattr(VaDER, "fit", forbid_fit)
    model = VaDER(
        n_steps=4,
        n_features=2,
        n_clusters=2,
        rnn_hidden_size=4,
        d_mu_stddev=2,
        batch_size=batch_size,
        epochs=1,
        pretrain_epochs=1,
        device="cpu",
        verbose=False,
    )
    model.model = _GaussianCore()
    X = np.zeros((n_samples, 4, 2), dtype=np.float32)
    expected = np.arange(n_samples) % 2
    X[:, :, 0] = np.where(expected[:, None] == 0, -1.0, 1.0)

    result = model.predict({"X": X}, return_latent_vars=return_latent_vars)

    assert result["clustering"].shape == (n_samples,)
    np.testing.assert_array_equal(result["clustering"], expected)
    if return_latent_vars:
        for key in ("mu_tilde", "stddev_tilde", "z", "imputation_latent"):
            assert result["latent_vars"][key].shape[0] == n_samples
    else:
        assert set(result) == {"clustering"}
        np.testing.assert_array_equal(model.cluster({"X": X}), expected)
