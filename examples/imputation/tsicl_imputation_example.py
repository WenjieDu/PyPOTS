"""
A minimalist, standalone example of the PyPOTS TS-ICL model for time-series imputation.
TS-ICL is a pretrained foundation model used zero-shot: there is no training step.

Note that the official pretrained TS-ICL weights, downloaded from Hugging Face on first use,
are licensed under the TS-ICL Non-Commercial License (non-commercial use only).
"""

import numpy as np
from benchpots.datasets import preprocess_random_walk
from pypots.imputation import TSICL
from pypots.nn.functional import calc_mse


def main():
    n_steps = 48
    n_features = 35

    # 1. Generate a random walk time-series dataset
    dataset = preprocess_random_walk(
        n_steps=n_steps, n_features=n_features, n_classes=5, n_samples_each_class=40, missing_rate=0.1
    )

    # 2. Extract the test set (no training set needed, TS-ICL is zero-shot)
    test_set = {"X": dataset["test_X"]}
    test_X_ori = dataset["test_X_ori"]

    # 3. Initialize the model (the pretrained checkpoint is downloaded from Hugging Face on first use)
    model = TSICL(device="cpu")

    # 4. Impute missing values
    print("🔮 Imputing missing values with the zero-shot TS-ICL model...")
    results = model.predict(test_set)
    imputed_X = results["imputation"]

    # 5. Evaluate on the artificially-masked values only
    indicating_mask = np.isnan(test_set["X"]) ^ np.isnan(test_X_ori)
    mse = calc_mse(imputed_X, np.nan_to_num(test_X_ori), indicating_mask)
    print(f"✅ The MSE of TS-ICL imputation is: {mse:.4f}")


if __name__ == "__main__":
    main()
