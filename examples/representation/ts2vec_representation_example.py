"""
A minimalist, standalone example of the PyPOTS TS2Vec model for time-series representation learning.
"""

import numpy as np
from benchpots.datasets import preprocess_random_walk

from pypots.representation import TS2Vec


def main():
    n_steps = 48
    n_features = 35
    n_output_dims = 16

    # 1. Generate a random walk time-series dataset with missing values
    dataset = preprocess_random_walk(
        n_steps=n_steps, n_features=n_features, n_classes=2, n_samples_each_class=40, missing_rate=0.1
    )

    # 2. Extract training, validation, and test sets (no labels needed)
    train_set = {"X": dataset["train_X"]}
    val_set = {"X": dataset["val_X"]}
    test_set = {"X": dataset["test_X"]}

    # 3. Initialize the model
    model = TS2Vec(
        n_steps=n_steps,
        n_features=n_features,
        n_output_dims=n_output_dims,
        d_hidden=64,
        n_layers=2,
        epochs=2,
        device="cpu",
    )

    # 4. Train the model
    print("🚀 Training the TS2Vec representation model...")
    model.fit(train_set, val_set)

    # 5. Generate timestamp-level and whole-series representations
    timestamp_repr = model.represent(test_set)
    series_repr = model.represent(test_set, encoding_window="full_series")

    # 6. Verify the output shapes and values
    n_samples = len(test_set["X"])
    assert timestamp_repr.shape == (n_samples, n_steps, n_output_dims)
    assert series_repr.shape == (n_samples, n_output_dims)
    assert np.isfinite(timestamp_repr).all() and np.isfinite(series_repr).all()

    print(f"✅ Timestamp-level representations: {timestamp_repr.shape}")
    print(f"✅ Whole-series representations: {series_repr.shape}")


if __name__ == "__main__":
    main()
