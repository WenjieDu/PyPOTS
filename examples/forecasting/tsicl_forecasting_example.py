"""
A minimalist, standalone example of the PyPOTS TS-ICL model for time-series forecasting.
TS-ICL is a pretrained foundation model used zero-shot: there is no training step.

Note that the official pretrained TS-ICL weights, downloaded from Hugging Face on first use,
are licensed under the TS-ICL Non-Commercial License (non-commercial use only).
"""

import numpy as np
from benchpots.datasets import preprocess_random_walk
from pypots.forecasting import TSICL
from pypots.nn.functional import calc_mse


def main():
    n_steps = 48
    n_pred_steps = 12
    n_features = 35

    # 1. Generate a random walk time-series dataset
    dataset = preprocess_random_walk(
        n_steps=n_steps + n_pred_steps, n_features=n_features, n_classes=5, n_samples_each_class=40, missing_rate=0.1
    )

    # 2. Extract the test set (no training set needed, TS-ICL is zero-shot)
    test_X = dataset["test_X"]
    test_set = {"X": test_X[:, :n_steps], "X_pred": test_X[:, n_steps:]}

    # 3. Initialize the model (the pretrained checkpoint is downloaded from Hugging Face on first use)
    model = TSICL(n_pred_steps=n_pred_steps, device="cpu")

    # 4. Forecast
    print("🔮 Forecasting future steps with the zero-shot TS-ICL model...")
    results = model.predict(test_set)
    forecasts = results["forecasting"]

    test_MSE = calc_mse(forecasts, np.nan_to_num(test_set["X_pred"]), ~np.isnan(test_set["X_pred"]))
    print(f"✅ TS-ICL forecasting MSE: {test_MSE:.4f}")


if __name__ == "__main__":
    main()
