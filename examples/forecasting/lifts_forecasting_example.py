"""Small, offline LIFTS forecasting example; inputs are already standardized."""

# Created by Sanyou Wu <sw4180@cumc.columbia.edu>
# License: BSD-3-Clause

import numpy as np

from pypots.forecasting import LIFTS

rng = np.random.default_rng(42)
sequence = rng.normal(size=(16, 8, 3)).astype(np.float32)
sequence[rng.random(sequence.shape) < 0.2] = np.nan
train_set = {"X": sequence[:12, :6], "X_pred": sequence[:12, 6:]}
val_set = {"X": sequence[12:, :6], "X_pred": sequence[12:, 6:]}
model = LIFTS(
    n_steps=6,
    n_features=3,
    n_pred_steps=2,
    d_latent=2,
    d_model=16,
    n_heads=2,
    d_ffn=32,
    n_layers=1,
    n_inner_layers=1,
    batch_size=8,
    epochs=2,
)
model.fit(train_set, val_set)
# Future targets are not supplied at prediction time.
result = model.predict({"X": sequence[12:, :6]}, n_sampling_times=10)
print("Point forecasts:", result["forecasting"].shape)  # (4, 2, 3)
print("Forecast samples:", result["forecasting_samples"].shape)  # (4, 10, 2, 3)
model.save("lifts_forecasting.pypots")
