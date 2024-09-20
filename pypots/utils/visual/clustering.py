"""
Utilities for clustering visualization.
"""

# Created by Bhargav Vemuri <vemuri.bhargav@gmail.com> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Dict

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


def get_cluster_members(test_data: np.ndarray, class_predictions: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Subset time series array using predicted cluster membership.

    Parameters
    __________
    test_data :
        Time series array that clusterer was run on.

    class_predictions:
        Clustering results returned by a clusterer.

    Returns
    _______
    cluster_members :
        Dictionary of test data time series organized by predicted cluster membership.
    """
    cluster_members = {}
    for i in np.unique(class_predictions):
        cluster_members[i] = test_data[class_predictions == i]
    return cluster_members


def clusters_for_plotting(
    cluster_members: Dict[int, np.ndarray],
) -> Dict[int, dict]:
    """
    Organize clustered arrays into format ready for plotting.

    Parameters
    __________
    cluster_members :
        Output from get_cluster_members function.

    Returns
    _______
    dict_to_plot :
        Test data organized by predicted cluster and time series variable.

        structure = { 'cluster0': {'var0': [['ts0'],['ts1'],...,['tsN']],
                                   'var1': [['ts0'],['ts1'],...,['tsN']],
                                   ...
                                   'varY': [['ts0'],['ts1'],...,['tsN']]
                                  },
                      ...,
                      'clusterX': {'var0': [['ts0'],['ts1'],...,['tsM']],
                                   'var1': [['ts0'],['ts1'],...,['tsM']],
                                   ...
                                   'varY': [['ts0'],['ts1'],...,['tsM']]
                                  }
                    }

                    where
                        clusterX is number of clusters predicted (n_clusters in model)
                        varY is number of time series variables recorded
                        tsN is number of members in cluster0, tsM is number of members in clusterX, etc.

    """
    dict_to_plot = {}

    for i in cluster_members:  # i iterates clusters
        dict_to_plot[i] = {}  # one dict per cluster
        for j in cluster_members[i]:  # j iterates members of each cluster
            temp = pd.DataFrame(j).to_dict(orient="list")  # dict of member's time series as lists (one per var)
            for key in temp:  # key is a time series var
                if key not in dict_to_plot[i]:
                    dict_to_plot[i][key] = [temp[key]]  # create entry in cluster dict for each time series var
                else:
                    dict_to_plot[i][key].append(temp[key])  # add cluster member's time series by var key
    return dict_to_plot


def plot_clusters(dict_to_plot: Dict[int, dict]) -> None:
    """
    Generate line plots of all cluster members per time series variable per cluster.

    Parameters
    __________
    dict_to_plot :
        Output from clusters_for_plotting function.
    """
    for i in dict_to_plot:  # iterate clusters
        for j in dict_to_plot[i]:  # iterate time series vars
            y = dict_to_plot[i][j]

            plt.figure(figsize=(16, 8))
            for y_values in y:  # iterate members
                x = np.arange(len(y_values))

                series1 = np.array(y_values).astype(np.double)
                s1mask = np.isfinite(
                    series1
                )  # connects all points (if >1) in line plot even if some intermediate are missing

                plt.plot(x[s1mask], series1[s1mask], ".-")

            plt.title("Cluster %i" % i)
            plt.ylabel("Var %i" % j)
            plt.xticks(x)
            plt.text(
                0.93,
                0.93,
                "n = %i" % len(y),
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=15,
            )
            plt.show()


def get_cluster_means(dict_to_plot: Dict[int, dict]) -> Dict[int, dict]:
    """
    Get time series variables' mean values and 95% confidence intervals at each time point per cluster.

    Parameters
    __________
    dict_to_plot :
        Output from clusters_for_plotting function.

    Returns
    _______
    cluster_means:
        Means and CI lower and upper bounds for each time series variable per cluster.

        structure = { 'var0': {'cluster0': {'mean': [tp0,tp1,...,tpN],
                                            'CI_low': [tp0,tp1,...tpN],
                                            'CI_high': [tp0,tp1,...tpN],
                                            'n': n0
                                            },
                               ...
                               'clusterX': {'mean': [tp0,tp1,...,tpN],
                                            'CI_low': [tp0,tp1,...tpN],
                                            'CI_high': [tp0,tp1,...tpN],
                                            'n': nX
                                            }
                              },
                      ...,
                      'varY': {'cluster0': {'mean': [tp0,tp1,...,tpN],
                                            'CI_low': [tp0,tp1,...tpN],
                                            'CI_high': [tp0,tp1,...tpN],
                                            'n': n0
                                            },
                               ...
                               'clusterX': {'mean': [tp0,tp1,...,tpN],
                                            'CI_low': [tp0,tp1,...tpN],
                                            'CI_high': [tp0,tp1,...tpN],
                                            'n': nX
                                            }
                              }
                    }

                    where
                        varY is number of time series variables recorded
                        clusterX is number of clusters predicted (n_clusters in model)
                        tpN is number of time points in each time series
                        n0 is the size of cluster0, nX is the size of clusterX, etc.


    """
    cluster_means = {}

    for i in dict_to_plot:  # iterate clusters
        for j in dict_to_plot[i]:  # iterate labs
            if j not in cluster_means:
                cluster_means[j] = {}

            cluster_means[j][i] = {}  # clusters nested within vars (reverse structure to clusters_for_plotting)

            cluster_means[j][i]["mean"] = list(
                pd.DataFrame(dict_to_plot[i][j]).mean(axis=0, skipna=True)
            )  # cluster mean array of time series var
            # CI calculation, from https://stackoverflow.com/a/34474255
            (cluster_means[j][i]["CI_low"], cluster_means[j][i]["CI_high"]) = st.t.interval(
                0.95,
                len(dict_to_plot[i][j]) - 1,  # degrees of freedom
                loc=cluster_means[j][i]["mean"],
                scale=pd.DataFrame(dict_to_plot[i][j]).sem(axis=0, skipna=True),
            )
            cluster_means[j][i]["n"] = len(dict_to_plot[i][j])  # save cluster size for downstream tasks/plotting

    return cluster_means


def plot_cluster_means(cluster_means: Dict[int, dict]) -> None:
    """
    Generate line plots of cluster means and 95% confidence intervals for each time series variable.

    Parameters
    __________
    cluster_means :
        Output from get_cluster_means function.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # to keep cluster colors consistent

    for i in cluster_means:  # iterate time series vars
        y = cluster_means[i]

        plt.figure(figsize=(16, 8))

        for y_values in y:  # iterate clusters
            for val in y[y_values]:  # iterate calculation (mean, CI_low, CI_high)
                if val == "mean":
                    x = np.arange(len(y[y_values][val]))
                    series1 = np.array(y[y_values][val]).astype(np.double)
                    s1mask = np.isfinite(series1)
                    plt.plot(
                        x[s1mask],
                        series1[s1mask],
                        ".-",  # mean as solid line
                        color=colors[y_values],
                        label="Cluster %i mean (n = %d)"
                        % (
                            y_values,
                            y[y_values]["n"],
                        ),  # legend will include cluster size
                    )

                if val in ("CI_low", "CI_high"):
                    x = np.arange(len(y[y_values][val]))
                    series1 = np.array(y[y_values][val]).astype(np.double)
                    s1mask = np.isfinite(series1)
                    plt.plot(
                        x[s1mask],
                        series1[s1mask],
                        "--",  # CI bounds as dashed lines
                        color=colors[y_values],
                    )

        plt.title("Var %d" % i)
        plt.xlabel("Timepoint")
        plt.xticks(x)

        # add dashed line label to legend
        line_dashed = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5, label="95% CI")
        handles, labels = plt.legend().axes.get_legend_handles_labels()
        handles.append(line_dashed)
        new_lgd = plt.legend(handles=handles)
        plt.gca().add_artist(new_lgd)

        plt.show()
