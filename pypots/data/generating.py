"""
Utilities for random data generating.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math
from typing import Optional, Tuple

import numpy as np
from pygrinder import mcar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


def gene_complete_random_walk(
    n_samples: int = 1000,
    n_steps: int = 24,
    n_features: int = 10,
    mu: float = 0.0,
    std: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate complete random walk time-series data, i.e. having no missing values.

    Parameters
    ----------
    n_samples : int, default=1000
        The number of training time-series samples to generate.

    n_steps: int, default=24
        The number of time steps (length) of generated time-series samples.

    n_features : int, default=10
        The number of features (dimensions) of generated time-series samples.

    mu : float, default=0.0
        Mean of the normal distribution, which random walk steps are sampled from.

    std : float, default=1.0
        Standard deviation of the normal distribution, which random walk steps are sampled from.

    random_state : int, default=None
        Random seed for data generation.

    Returns
    -------
    ts_samples: array, shape of [n_samples, n_steps, n_features]
        Generated random walk time series.
    """
    seed = check_random_state(random_state)
    ts_samples = np.zeros([n_samples, n_steps, n_features])
    random_values = seed.randn(n_samples, n_steps, n_features) * std + mu
    ts_samples[:, 0, :] = random_values[:, 0, :]
    for t in range(1, n_steps):
        ts_samples[:, t, :] = ts_samples[:, t - 1, :] + random_values[:, t, :]
    ts_samples = np.asarray(ts_samples)
    return ts_samples


def gene_complete_random_walk_for_classification(
    n_classes: int = 2,
    n_samples_each_class: int = 500,
    n_steps: int = 24,
    n_features: int = 10,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate complete random walk time-series data for the classification task.

    Parameters
    ----------
    n_classes : int, must >=1, default=2
        Number of classes (types) of the generated data.

    n_samples_each_class : int, default=500
        Number of samples for each class to generate.

    n_steps : int, default=24
        Number of time steps in each sample.

    n_features : int, default=10
        Number of features.

    shuffle : bool, default=True
        Whether to shuffle generated samples.
        If not, you can separate samples of each class according to `n_samples_each_class`.
        For example,
        X_class0=X[:n_samples_each_class],
        X_class1=X[n_samples_each_class:n_samples_each_class*2]

    random_state : int, default=None
        Random seed for data generation.

    Returns
    -------
    X : array, shape of [n_samples, n_steps, n_features]
        Generated time-series data.

    y : array, shape of [n_samples]
        Labels indicating classes of time-series samples.

    """
    assert n_classes > 1, f"n_classes should be >1, but got {n_classes}"

    ts_collector = []
    label_collector = []

    mu = 0
    std = 1

    for c_ in range(n_classes):
        ts_samples = gene_complete_random_walk(
            n_samples_each_class, n_steps, n_features, mu, std, random_state
        )
        label_samples = np.asarray([1 for _ in range(n_samples_each_class)]) * c_
        ts_collector.extend(ts_samples)
        label_collector.extend(label_samples)
        mu += 1

    X = np.asarray(ts_collector)
    y = np.asarray(label_collector)

    # if shuffling, then shuffle the order of samples
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    return X, y


def gene_complete_random_walk_for_anomaly_detection(
    n_samples: int = 1000,
    n_steps: int = 24,
    n_features: int = 10,
    mu: float = 0.0,
    std: float = 1.0,
    anomaly_proportion: float = 0.1,
    anomaly_fraction: float = 0.02,
    anomaly_scale_factor: float = 2.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random walk time-series data for the anomaly-detection task.

    Parameters
    ----------
    n_samples : int, default=1000
        The number of training time-series samples to generate.

    n_features : int, default=10
        The number of features (dimensions) of generated time-series samples.

    n_steps: int, default=24
        The number of time steps (length) of generated time-series samples.

    mu : float, default=0.0
        Mean of the normal distribution, which random walk steps are sampled from.

    std : float, default=1.0
        Standard deviation of the normal distribution, which random walk steps are sampled from.

    anomaly_proportion : float, default=0.1
        Proportion of anomaly samples in all samples.

    anomaly_fraction : float, default=0.02
        Fraction of anomaly points in each anomaly sample.

    anomaly_scale_factor : float, default=2.0
        Scale factor for value scaling to create anomaly points in time series samples.

    random_state : int, default=None
        Random seed for data generation.

    Returns
    -------
    X : array, shape of [n_samples, n_steps, n_features]
        Generated time-series data.

    y : array, shape of [n_samples]
        Labels indicating if time-series samples are anomalies.
    """
    assert (
        0 < anomaly_proportion < 1
    ), f"anomaly_proportion should be >0 and <1, but got {anomaly_proportion}"
    assert (
        0 < anomaly_fraction < 1
    ), f"anomaly_fraction should be >0 and <1, but got {anomaly_fraction}"
    seed = check_random_state(random_state)
    X = seed.randn(n_samples, n_steps, n_features) * std + mu
    n_anomaly = math.floor(n_samples * anomaly_proportion)
    anomaly_indices = np.random.choice(n_samples, size=n_anomaly, replace=False)
    for a_i in anomaly_indices:
        anomaly_sample = X[a_i]
        anomaly_sample = anomaly_sample.flatten()
        min_val = anomaly_sample.min()
        max_val = anomaly_sample.max()
        max_difference = min_val - max_val
        n_points = n_steps * n_features
        n_anomaly_points = int(n_points * anomaly_fraction)
        point_indices = np.random.choice(
            a=n_points, size=n_anomaly_points, replace=False
        )
        for p_i in point_indices:
            anomaly_sample[p_i] = mu + np.random.uniform(
                low=min_val - anomaly_scale_factor * max_difference,
                high=max_val + anomaly_scale_factor * max_difference,
            )
        X[a_i] = anomaly_sample.reshape(n_steps, n_features)

    # create labels
    y = np.zeros(n_samples)
    y[anomaly_indices] = 1

    # shuffling
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y


def gene_random_walk(
    n_steps=24,
    n_features=10,
    n_classes=2,
    n_samples_each_class=1000,
    missing_rate=0.1,
) -> dict:
    """Generate a random-walk data.

    Parameters
    ----------
    n_steps : int, default=24
        Number of time steps in each sample.

    n_features : int, default=10
        Number of features.

    n_classes : int, default=2
        Number of classes (types) of the generated data.

    n_samples_each_class : int, default=1000
        Number of samples for each class to generate.

    missing_rate : float, default=0.1
        The rate of randomly missing values to generate, should be in [0,1).

    Returns
    -------
    data: dict,
        A dictionary containing the generated data.
    """
    assert 0 <= missing_rate < 1, "missing_rate must be in [0,1)"

    # generate samples
    X, y = gene_complete_random_walk_for_classification(
        n_classes=n_classes,
        n_samples_each_class=n_samples_each_class,
        n_steps=n_steps,
        n_features=n_features,
    )
    # split into train/val/test sets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)

    if missing_rate > 0:
        # create random missing values
        train_X_ori = train_X
        train_X = mcar(train_X, missing_rate)
        # test set is left to mask after normalization

    train_X = train_X.reshape(-1, n_features)
    val_X = val_X.reshape(-1, n_features)
    test_X = test_X.reshape(-1, n_features)
    # normalization
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    # reshape into time series samples
    train_X = train_X.reshape(-1, n_steps, n_features)
    val_X = val_X.reshape(-1, n_steps, n_features)
    test_X = test_X.reshape(-1, n_steps, n_features)
    data = {
        "n_classes": n_classes,
        "n_steps": n_steps,
        "n_features": n_features,
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "test_X": test_X,
        "test_y": test_y,
        "scaler": scaler,
    }

    if missing_rate > 0:
        # mask values in the test set as ground truth
        train_X_ori = scaler.transform(train_X_ori.reshape(-1, n_features)).reshape(
            -1, n_steps, n_features
        )
        data["train_X_ori"] = train_X_ori

        val_X_ori = val_X
        val_X = mcar(val_X, missing_rate)
        data["val_X"] = val_X
        data["val_X_ori"] = val_X_ori

        test_X_ori = test_X
        test_X = mcar(test_X, missing_rate)
        data["test_X"] = test_X
        data["test_X_ori"] = np.nan_to_num(test_X_ori)  # fill NaNs for later error calc
        data["test_X_indicating_mask"] = np.isnan(test_X_ori) ^ np.isnan(test_X)

    return data
