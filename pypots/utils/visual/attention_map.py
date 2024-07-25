"""
Utilities for attention map visualization.
"""

# Created by Anshuman Swain <aswai@seas.upenn.edu> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

try:
    import seaborn as sns
except Exception:
    pass


def plot_attention(timeSteps: ArrayLike, attention: np.ndarray, fontscale=None):
    """Visualize the map of attention weights from Transformer-based models.

    Parameters
    ---------------
    timeSteps: 1D array-like object, preferable list of strings
        A vector containing the time steps of the input.
        The time steps will be converted to a list of strings if they are not already.

    attention: 2D array-like object
        A 2D matrix representing the attention weights

    fontscale: float/int
        Sets the scale for fonts in the Seaborn heatmap (applied to sns.set_theme(font_scale = _)


    Return
    ---------------
    ax: Matplotlib axes object

    """

    if not all(isinstance(ele, str) for ele in timeSteps):
        timeSteps = [str(step) for step in timeSteps]

    if fontscale is not None:
        sns.set_theme(font_scale=fontscale)

    fig, ax = plt.subplots()
    ax.tick_params(left=True, bottom=True, labelsize=10)
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])

    assert attention.ndim == 2, "The attention matrix is not two-dimensional"
    sns.heatmap(
        attention,
        ax=ax,
        xticklabels=timeSteps,
        yticklabels=timeSteps,
        linewidths=0,
        cbar=True,
    )
    cb = ax.collections[0].colorbar
    cb.ax.tick_params(labelsize=10)

    return fig
