"""
Utilities for data visualization.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..logging import logger


def plot_data(
    X: np.ndarray,
    X_ori: np.ndarray,
    X_imputed: np.ndarray,
    sample_idx: Optional[int] = None,
    n_rows: int = 10,
    n_cols: int = 4,
    fig_size: Optional[list] = None,
):
    """Plot the imputed values, the observed values, and the evaluated values of one multivariate timeseries.
    The observed values are marked with red 'x',  the evaluated values are marked with blue 'o',
    and the imputed values are marked with solid green line.

    Parameters
    ----------
    X :
        The observed values

    X_ori :
        The evaluated values

    X_imputed :
        The imputed values

    sample_idx :
        The index of the sample to be plotted.
        If None, a randomly-selected sample will be plotted for visualization.

    n_rows :
        The number of rows in the plot

    n_cols :
        The number of columns in the plot

    fig_size :
        The size of the figure
    """

    vals_shape = X.shape
    assert len(vals_shape) == 3, "vals_obs should be a 3D array of shape (n_samples, n_steps, n_features)"
    n_samples, n_steps, n_features = vals_shape

    if sample_idx is None:
        sample_idx = np.random.randint(low=0, high=n_samples)
        logger.warning(f"⚠️ No sample index is specified, a random sample {sample_idx} is selected for visualization.")

    if fig_size is None:
        fig_size = [24, 36]

    n_k = n_rows * n_cols
    K = np.min([n_features, n_k])
    L = n_steps
    plt.rcParams["font.size"] = 16
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_size[0], fig_size[1]))

    for k in range(K):
        df = pd.DataFrame({"x": np.arange(0, L), "val": X_imputed[sample_idx, :, k]})
        df1 = pd.DataFrame({"x": np.arange(0, L), "val": X[sample_idx, :, k]})
        df2 = pd.DataFrame({"x": np.arange(0, L), "val": X_ori[sample_idx, :, k]})
        row = k // n_cols
        col = k % n_cols
        axes[row][col].plot(df1.x, df1.val, color="r", marker="x", linestyle="None")
        axes[row][col].plot(df2.x, df2.val, color="b", marker="o", linestyle="None")
        axes[row][col].plot(df.x, df.val, color="g", linestyle="solid")
        if col == 0:
            plt.setp(axes[row, 0], ylabel="value")
        if row == -1:
            plt.setp(axes[-1, col], xlabel="time")

    logger.info("Plotting finished. Please invoke matplotlib.pyplot.show() to display the plot.")


def plot_missingness(
    missing_mask: int,
    min_step: int = 0,
    max_step: int = 1,
    sample_idx: Optional[int] = None,
):
    """Plot the missingness pattern of one multivariate timeseries. For each feature,
    the observed timestamp is marked with blue '|'. The distribution of sequence lengths is also plotted.
    Hereby, the sequence length is defined as the number of observed timestamps in one feature.

    Parameters
    ----------
    missing_mask :
        The missing mask of multivariate time series.

    min_step :
        The minimum time step for visualization.

    max_step :
        The maximum time step for visualization.

    sample_idx :
        The index of the sample to be plotted, if None, a randomly-selected sample will be plotted for visualization.
    """
    mask_shape = missing_mask.shape
    if len(mask_shape) == 3:
        n_samples, n_steps, n_features = missing_mask.shape
        if sample_idx is None:
            sample_idx = np.random.randint(low=0, high=n_samples)
            logger.warning(
                f"⚠️ No sample index is specified, a random sample {sample_idx} is selected for visualization."
            )
        mask_sample_for_vis = np.transpose(missing_mask[sample_idx], (1, 0))
    elif len(mask_shape) == 2:
        n_steps, n_features = missing_mask.shape
        mask_sample_for_vis = np.transpose(missing_mask, (1, 0))
    else:
        raise ValueError(
            f"missing_mask should be missing masks of multiple time series samples of "
            f"shape (n_samples, n_steps, n_features), "
            f"or of a single time series sample of shape (n_steps, n_features). "
            f"But got invalid shape of missing_mask: {mask_shape}."
        )

    time = np.repeat(
        np.linspace(0, max_step, n_steps).reshape(n_steps, 1),
        axis=1,
        repeats=n_features,
    )
    plot_sample = np.transpose(time, (1, 0))

    fig, axes = plt.subplots(figsize=[12, 3.5], dpi=200, nrows=1, ncols=2)
    plt.subplots_adjust(hspace=0.1)
    seq_len = []
    for feature_idx in range(n_features):
        t = plot_sample[feature_idx][mask_sample_for_vis[feature_idx] == 1]
        axes[0].scatter(t, np.ones_like(t) * feature_idx, alpha=1, c="C0", marker="|")
        seq_len.append(len(t))
    axes[0].set_title("Visualization of arrival times", fontsize=9)
    axes[0].set_xlabel("Time", fontsize=7)
    axes[0].set_ylabel("Features #", fontsize=7)
    axes[0].set_xlim(-1, n_steps)
    # axes[0].set_ylim(0, n_features-1)
    axes[0].tick_params(axis="both", labelsize=7)

    axes[1].set_title("Distribution of sequence lengths", fontsize=9)
    axes[1].hist(
        seq_len,
        bins=n_steps,
        color="C1",
        range=(min_step, max_step),
    )
    axes[1].set_xlabel(r"Sequence length", fontsize=7)
    axes[1].set_ylabel("Frequency", fontsize=7)
    axes[1].tick_params(axis="both", labelsize=7)

    logger.info("Plotting finished. Please invoke matplotlib.pyplot.show() to display the plot.")
