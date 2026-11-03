"""Contract and edge-case tests for the LIFTS imputer (no external datasets)."""

import h5py
import numpy as np
import pytest
import torch

from pypots.imputation import LIFTS
from pypots.nn.modules.lifts import BackboneLIFTS

pytestmark = pytest.mark.xdist_group(name="imputation-lifts")


def make_model(**kwargs):
    return LIFTS(
        n_steps=4,
        n_features=3,
        d_latent=2,
        d_model=8,
        n_heads=2,
        d_ffn=16,
        n_layers=1,
        n_inner_layers=1,
        epochs=1,
        batch_size=2,
        device="cpu",
        verbose=False,
        **kwargs,
    )


def make_data():
    rng = np.random.default_rng(42)
    original = rng.normal(size=(5, 4, 3)).astype(np.float32)
    original[:, 0, 0] = np.nan  # genuine missing values have no evaluation target
    X = original.copy()
    X[:, 1, 1] = np.nan  # held-out observed entries for validation
    return {"X": X, "X_ori": original}


def test_fit_samples_and_observed_values():
    torch.manual_seed(42)
    data = make_data()
    model = make_model()
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    model.fit({"X": data["X"]}, data)
    result = model.predict(data, n_sampling_times=3)
    assert result["imputation"].shape == (5, 4, 3)
    assert result["imputation_samples"].shape == (5, 3, 4, 3)
    assert np.isfinite(result["imputation_samples"]).all()
    np.testing.assert_allclose(result["imputation"], result["imputation_samples"].mean(axis=1))
    observed = ~np.isnan(data["X"])
    expected = np.broadcast_to(data["X"][:, None], result["imputation_samples"].shape)
    np.testing.assert_array_equal(
        result["imputation_samples"][np.broadcast_to(observed[:, None], expected.shape)],
        expected[np.broadcast_to(observed[:, None], expected.shape)],
    )
    assert result["imputation_samples"][:, :, 1, 1].std(axis=1).max() > 0
    assert np.isfinite(model.best_loss)
    assert any(not torch.equal(before[k], v) for k, v in model.model.state_dict().items())
    assert model.impute(data).shape == (5, 4, 3)


def test_hdf5_save_load_and_edge_masks(tmp_path):
    data = make_data()
    path = str(tmp_path / "data.h5")
    with h5py.File(path, "w") as handle:
        for k, value in data.items():
            handle[k] = value
    model = make_model(saving_path=str(tmp_path / "logs"))
    model.fit(path, path)
    assert np.isfinite(model.impute(path)).all()
    checkpoint = str(tmp_path / "model.pypots")
    model.save(checkpoint)
    restored = make_model()
    restored.load(checkpoint)
    torch.manual_seed(7)
    expected = model.predict(data, n_sampling_times=2)
    torch.manual_seed(7)
    actual = restored.predict(data, n_sampling_times=2)
    np.testing.assert_array_equal(actual["imputation_samples"], expected["imputation_samples"])
    for X in [np.zeros((1, 4, 3), dtype=np.float32), np.full((1, 4, 3), np.nan, dtype=np.float32)]:
        assert np.isfinite(restored.impute({"X": X})).all()
    assert list((tmp_path / "logs").rglob("*.pypots"))
    assert list((tmp_path / "logs").rglob("events.out.tfevents.*"))


def test_filter_is_causal_and_masks_are_informative():
    model = BackboneLIFTS(4, 3, 2, 8, 2, 16, 1, 1).eval()
    X = torch.randn(2, 4, 3)
    mask = torch.ones_like(X)
    mask[:, 0, 0] = 0
    changed = X.clone()
    changed[:, 2:] += 10
    torch.manual_seed(12)
    first = model.encoder_h(X, mask)
    torch.manual_seed(12)
    second = model.encoder_h(changed, mask)
    torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)
    other_mask = mask.clone()
    other_mask[:, 1, 1] = 0
    torch.manual_seed(12)
    third = model.encoder_h(X, other_mask)
    assert not torch.equal(first[:, 1], third[:, 1])


@pytest.mark.parametrize(
    "kwargs", [{"d_model": 6, "n_heads": 2}, {"self_mask_rate": 0}, {"d_latent": 0}, {"n_sampling_times_train": 0}]
)
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        LIFTS(n_steps=4, n_features=3, device="cpu", verbose=False, **kwargs)


def test_invalid_input_and_sampling():
    model = make_model()
    with pytest.raises(ValueError, match="shape"):
        model.predict({"X": np.zeros((1, 5, 3), dtype=np.float32)})
    for n in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="positive integer"):
            model.predict(make_data(), n_sampling_times=n)
    with pytest.raises(ValueError, match="X_ori"):
        model.fit(make_data(), {"X": make_data()["X"]})
