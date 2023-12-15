"""
Utilities for data visualization.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_data(
    vals_obs,
    vals_eval,
    vals_imputed,
    dataidx: int = None,
    nrows: int = 10,
    ncols: int = 4,
    figsize=None,
):
    """Plot the imputed values, the observed values, and the evaluated values of one multivariate timeseries.
    The observed values are marked with red 'x',  the evaluated values are marked with blue 'o',
    and the imputed values are marked with solid green line.

    Parameters
    ----------
    vals_obs : ndarray,
        The observed values

    vals_eval : ndarray,
        The evaluated values

    vals_imputed : ndarray,
        The imputed values

    dataidx : int,
        The index of the sample to be plotted

    nrows : int,
        The number of rows in the plot

    ncols : int,
        The number of columns in the plot

    figsize : list,
        The size of the figure
    """

    n_samples, n_steps, n_features = vals_obs.shape

    if dataidx is None:
        dataidx = np.random.randint(low=0, high=n_samples)
    if figsize is None:
        figsize = [24, 36]

    n_k = nrows * ncols
    K = np.min([n_features, n_k])
    L = n_steps
    plt.rcParams["font.size"] = 16
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0], figsize[1]))
    # fig.delaxes(axes[-1][-1])

    for k in range(K):
        df = pd.DataFrame({"x": np.arange(0, L), "val": vals_imputed[dataidx, :, k]})
        df1 = pd.DataFrame({"x": np.arange(0, L), "val": vals_obs[dataidx, :, k]})
        df2 = pd.DataFrame({"x": np.arange(0, L), "val": vals_eval[dataidx, :, k]})
        row = k // ncols
        col = k % ncols
        axes[row][col].plot(df1.x, df1.val, color="r", marker="x", linestyle="None")
        axes[row][col].plot(df2.x, df2.val, color="b", marker="o", linestyle="None")
        axes[row][col].plot(df.x, df.val, color="g", linestyle="solid")
        if col == 0:
            plt.setp(axes[row, 0], ylabel="value")
        if row == -1:
            plt.setp(axes[-1, col], xlabel="time")


def plot_missingness(mask, t_max=1, t_min=0, dataidx=None):
    """Plot the missingness pattern of one multivariate timeseries. For each feature,
    the observed timestamp is marked with blue '|'. The distribution of sequence lengths is also plotted.
    Hereby, the sequence length is defined as the number of observed timestamps in one feature.

    Parameters
    ----------
    mask : ndarray,
        The mask matrix of one multivariate timeseries

    t_max : int,
        The maximum time

    t_min : int,
        The minimum time

    dataidx : int,
        The index of the sample to be plotted
    """
    n_s, n_l, n_c = mask.shape
    time = np.repeat(
        np.repeat(np.linspace(0, t_max, n_l).reshape(1, n_l, 1), axis=2, repeats=n_c),
        axis=0,
        repeats=n_s,
    )
    if dataidx is None:
        dataidx = np.random.randint(low=0, high=n_s)

    fig, axes = plt.subplots(figsize=[12, 3.5], dpi=200, nrows=1, ncols=2)
    plt.subplots_adjust(hspace=0.1)
    seq_len = []
    sample = np.transpose(time[dataidx], (1, 0))
    mask_s = np.transpose(mask[dataidx], (1, 0))
    for feature_idx in range(n_c):
        t = sample[feature_idx][mask_s[feature_idx] == 1]
        axes[0].scatter(t, np.ones_like(t) * (feature_idx), alpha=1, c="C0", marker="|")
        seq_len.append(len(t))
    axes[0].set_title("Visualization of arrival times", fontsize=9)
    axes[0].set_xlabel("Time", fontsize=7)
    axes[0].set_ylabel("Features #", fontsize=7)
    axes[0].set_xlim(-1, n_l)
    # axes[0].set_ylim(0, n_c-1)
    axes[0].tick_params(axis="both", labelsize=7)

    axes[1].set_title("Distribution of sequence lengths", fontsize=9)
    axes[1].hist(
        seq_len,
        bins=n_l,
        color="C1",
        range=(t_min, t_max),
    )
    axes[1].set_xlabel(r"Sequence length", fontsize=7)
    axes[1].set_ylabel("Frequency", fontsize=7)
    axes[1].tick_params(axis="both", labelsize=7)
    plt.show()
