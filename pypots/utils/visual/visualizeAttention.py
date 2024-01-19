import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import numppy as np


def visualizeAttention(timeSteps: ArrayLike, attention: np.ndarray):
    """Visualize the attention matrix

    Parameters
    ---------------
    timeSteps: 1D array-like object, preferable list of strings
        A vector containing the time steps of the input. The time steps will be converted to a list of strings if they are not already.

    attention: 2D array-like object
        A 2D matrix representing the attention weights


    Return
    ---------------
    ax: Matplotlib axes object

    """

    if not all(isinstance(ele, str) for ele in timeSteps):
        timeSteps = [str(timeSteps) for step in timeSteps]

    fig, ax = plt.subplots()
    ax.tickparams(left=True, bottom=True, labelsize=10)
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

    return ax
