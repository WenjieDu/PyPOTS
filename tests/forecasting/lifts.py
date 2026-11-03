"""Contract and future-data isolation tests for LIFTS forecasting."""

import h5py
import numpy as np
import pytest
import torch

from pypots.forecasting import LIFTS

pytestmark = pytest.mark.xdist_group(name="forecasting-lifts")


def make_model(**kwargs):
    return LIFTS(
        n_steps=4,
        n_features=3,
        n_pred_steps=2,
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
    rng = np.random.default_rng(43)
    X = rng.normal(size=(5, 4, 3)).astype(np.float32)
    future = rng.normal(size=(5, 2, 3)).astype(np.float32)
    X[:, 1, 1] = np.nan
    future[:, 0, 1] = np.nan
    return {"X": X, "X_pred": future}


def test_fit_forecast_samples_and_future_isolation():
    torch.manual_seed(42)
    data = make_data()
    model = make_model()
    model.fit(data, data)
    torch.manual_seed(5)
    result = model.predict({"X": data["X"]}, n_sampling_times=3)
    assert result["forecasting"].shape == (5, 2, 3)
    assert result["forecasting_samples"].shape == (5, 3, 2, 3)
    assert np.isfinite(result["forecasting_samples"]).all()
    assert result["forecasting_samples"].std(axis=1).max() > 0
    np.testing.assert_allclose(result["forecasting"], result["forecasting_samples"].mean(axis=1))
    # Prediction must ignore any supplied future targets.
    torch.manual_seed(5)
    poisoned = model.predict({"X": data["X"], "X_pred": np.full_like(data["X_pred"], 1e6)}, n_sampling_times=3)
    np.testing.assert_array_equal(result["forecasting_samples"], poisoned["forecasting_samples"])
    assert model.forecast({"X": data["X"]}).shape == (5, 2, 3)
    assert np.isfinite(model.best_loss)


def test_file_input_and_checkpoint(tmp_path):
    data = make_data()
    path = str(tmp_path / "forecast.h5")
    with h5py.File(path, "w") as handle:
        for k, value in data.items():
            handle[k] = value
    model = make_model()
    model.fit(path, path)
    assert np.isfinite(model.forecast(path)).all()
    checkpoint = str(tmp_path / "model.pypots")
    model.save(checkpoint)
    restored = make_model()
    restored.load(checkpoint)
    torch.manual_seed(6)
    expected = model.predict(data, n_sampling_times=2)
    torch.manual_seed(6)
    actual = restored.predict(data, n_sampling_times=2)
    np.testing.assert_array_equal(expected["forecasting_samples"], actual["forecasting_samples"])


def test_invalid_target_shape_and_sampling():
    model = make_model()
    data = make_data()
    with pytest.raises(ValueError, match="X_pred"):
        model.fit({"X": data["X"], "X_pred": np.zeros((5, 3, 3), dtype=np.float32)})
    with pytest.raises(ValueError, match="positive integer"):
        model.predict({"X": data["X"]}, n_sampling_times=0)
    with pytest.raises(ValueError, match="n_pred_steps"):
        LIFTS(n_steps=4, n_features=3, n_pred_steps=0)
