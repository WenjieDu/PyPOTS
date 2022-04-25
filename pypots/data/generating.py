"""
Utilities for random data generating.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import math

import numpy as np
from sklearn.utils import check_random_state


def generate_random_walk(n_samples=1000, n_steps=24, n_features=10, mu=0., std=1., random_state=None):
    """ Generate random walk time-series data.

    Parameters
    ----------
    n_samples : int, default=1000
        The number of training time-series samples to generate.
    n_steps: int, default=24
        The number of time steps (length) of generated time-series samples.
    n_features : int, default=10
        The number of features (dimensions) of generated time-series samples.
    mu : float, default 0.0,
        Mean of the normal distribution, which random walk steps are sampled from.
    std : float, default 1.,
        Standard deviation of the normal distribution, which random walk steps are sampled from.
    random_state : int or numpy.RandomState, default=None,
        Random seed for data generation.

    Returns
    -------
    array, shape of [n_samples, n_steps, n_features]
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


def generate_random_walk_for_classification(n_classes=2, n_samples_each_class=500, n_steps=24, n_features=10,
                                            shuffle=True, random_state=None):
    """ Generate random walk time-series data for the classification task.

    Parameters
    ----------
    n_classes : int, default=2
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
    random_state : int or numpy.RandomState, default=None,
        Random seed for data generation.

    Returns
    -------
    X : array, shape of [n_classes*n_samples_each_class, n_steps, n_features]
        Generated time-series data.
    y : array, shape of [n_classes*n_samples_each_class]
        Labels indicating classes of time-series samples.

    """
    ts_collector = []
    label_collector = []

    mu = 0
    std = 1

    for c_ in range(n_classes):
        ts_samples = generate_random_walk(n_samples_each_class, n_steps, n_features, mu, std, random_state)
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


def generate_random_walk_for_anomaly_detection(n_samples=1000, n_steps=24, n_features=10, mu=0., std=1.,
                                               anomaly_proportion=0.1, anomaly_fraction=0.02, anomaly_scale_factor=2.0,
                                               random_state=None):
    """ Generate random walk time-series data for the anomaly-detection task.

    Parameters
    ----------
    n_samples : int, default=1000
        The number of training time-series samples to generate.
    n_features : int, default=10
        The number of features (dimensions) of generated time-series samples.
    n_steps: int, default=24
        The number of time steps (length) of generated time-series samples.
    mu : float, default 0.0,
        Mean of the normal distribution, which random walk steps are sampled from.
    std : float, default 1.,
        Standard deviation of the normal distribution, which random walk steps are sampled from.
    anomaly_proportion : float, in (0,1)
        Proportion of anomaly samples in all samples.
    anomaly_fraction : float, in (0,1)
        Fraction of anomaly points in each anomaly sample.
    anomaly_scale_factor : int or float,
        Scale factor for value scaling to create anomaly points in time series samples.
    random_state : int or numpy.RandomState, default=None,
        Random seed for data generation.

    Returns
    -------
    X : array, shape of [n_classes*n_samples_each_class, n_steps, n_features]
        Generated time-series data.
    y : array, shape of [n_classes*n_samples_each_class]
        Labels indicating if time-series samples are anomalies.
    """
    assert 0 < anomaly_proportion < 1, f'anomaly_proportion should be >0 and <1, but got {anomaly_proportion}'
    assert 0 < anomaly_fraction < 1, f'anomaly_fraction should be >0 and <1, but got {anomaly_fraction}'
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
        point_indices = np.random.choice(a=n_points, size=n_anomaly_points, replace=False)
        for p_i in point_indices:
            anomaly_sample[p_i] = mu + np.random.uniform(low=min_val - anomaly_scale_factor * max_difference,
                                                         high=max_val + anomaly_scale_factor * max_difference)
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
