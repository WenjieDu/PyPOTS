"""
Utilities for random data generating.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import numpy as np
from sklearn.utils import check_random_state


def generate_random_walk(n_samples=1000, n_steps=24, n_features=10, mu=0., std=1., random_state=None):
    """
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
    random_state : int or numpy.RandomState, default=None,
        Random seed for data generation.

    Returns
    -------
    array, shape of [n_samples, n_steps, n_features]
        Generated random walk time series.
    """
    seed = check_random_state(random_state)
    ts_samples = np.empty((n_samples, n_steps, n_features))
    noise = seed.randn(n_samples, n_steps, n_features) * std + mu
    ts_samples[:, 0, :] = noise[:, 0, :]
    for t in range(1, n_steps):
        ts_samples[:, t, :] = ts_samples[:, t - 1, :] + noise[:, t, :]
    ts_samples = np.asarray(ts_samples)
    return ts_samples


def generate_random_walk_for_classification(n_classes=2, n_samples_each_class=500, n_steps=24, n_features=10,
                                            random_state=None):
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
        std += 1

    ts_collector = np.asarray(ts_collector)
    label_collector = np.asarray(label_collector)
    return ts_collector, label_collector
