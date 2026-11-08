"""Small, offline LIFTS imputation example; inputs are already standardized."""

# Created by Sanyou Wu <sw4180@cumc.columbia.edu>
# License: BSD-3-Clause

import numpy as np

from pypots.imputation import LIFTS

rng = np.random.default_rng(42)
X_ori = rng.normal(size=(16, 6, 3)).astype(np.float32)
X = X_ori.copy()
X[rng.random(X.shape) < 0.2] = np.nan
model = LIFTS(
    n_steps=6,
    n_features=3,
    d_latent=2,
    d_model=16,
    n_heads=2,
    d_ffn=32,
    n_layers=1,
    n_inner_layers=1,
    batch_size=8,
    epochs=2,
)
model.fit({"X": X[:12]}, {"X": X[12:], "X_ori": X_ori[12:]})
result = model.predict({"X": X[12:]}, n_sampling_times=10)
print("Point imputation:", result["imputation"].shape)  # (4, 6, 3)
print("Multiple imputations:", result["imputation_samples"].shape)  # (4, 10, 6, 3)
# The same fitted model also provides the original in-sample one-step forecasts.
one_step = model.forecast_in_sample({"X": X[12:]})  # 100 draws by default
print("One-step forecasts:", one_step["forecasting"].shape)  # (4, 6, 3)
print("One-step samples:", one_step["forecasting_samples"].shape)  # (4, 100, 6, 3)
model.save("lifts_imputation.pypots")
