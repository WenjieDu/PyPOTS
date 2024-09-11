"""
Evaluation metrics related to clustering.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
from sklearn import metrics


def calc_rand_index(
    class_predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculate Rand Index, a measure of the similarity between two data clusterings.
    Refer to :cite:`rand1971RandIndex`.

    Parameters
    ----------
    class_predictions :
        Clustering results returned by a clusterer.

    targets :
        Ground truth (correct) clustering results.

    Returns
    -------
    RI :
        Rand index.

    References
    ----------
    .. L. Hubert and P. Arabie, Comparing Partitions, Journal of
      Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075

    .. https://en.wikipedia.org/wiki/Simple_matching_coefficient

    .. https://en.wikipedia.org/wiki/Rand_index
    """
    # # detailed implementation
    # n = len(targets)
    # TP = 0
    # TN = 0
    # for i in range(n - 1):
    #     for j in range(i + 1, n):
    #         if targets[i] != targets[j]:
    #             if class_predictions[i] != class_predictions[j]:
    #                 TN += 1
    #         else:
    #             if class_predictions[i] == class_predictions[j]:
    #                 TP += 1
    #
    # RI = n * (n - 1) / 2
    # RI = (TP + TN) / RI

    RI = metrics.rand_score(targets, class_predictions)

    return RI


def calc_adjusted_rand_index(
    class_predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculate adjusted Rand Index.

    Parameters
    ----------
    class_predictions :
        Clustering results returned by a clusterer.

    targets :
        Ground truth (correct) clustering results.

    Returns
    -------
    aRI :
        Adjusted Rand index.

    References
    ----------
    .. [1] `L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      <https://link.springer.com/article/10.1007%2FBF01908075>`_

    .. [2] `D. Steinley, Properties of the Hubert-Arabie
      adjusted Rand index, Psychological Methods 2004
      <https://psycnet.apa.org/record/2004-17801-007>`_

    .. [3] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

    """
    aRI = metrics.adjusted_rand_score(targets, class_predictions)
    return aRI


def calc_nmi(
    class_predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculate Normalized Mutual Information between two clusterings.

    Parameters
    ----------
    class_predictions :
        Clustering results returned by a clusterer.

    targets :
        Ground truth (correct) clustering results.

    Returns
    -------
    NMI : float,
        Normalized Mutual Information


    """
    NMI = metrics.normalized_mutual_info_score(targets, class_predictions)
    return NMI


def calc_cluster_purity(
    class_predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculate cluster purity.

    Parameters
    ----------
    class_predictions :
        Clustering results returned by a clusterer.

    targets :
        Ground truth (correct) clustering results.

    Returns
    -------
    cluster_purity :
        cluster purity.

    Notes
    -----
    This function is from the answer https://stackoverflow.com/a/51672699 on StackOverflow.

    """
    contingency_matrix = metrics.cluster.contingency_matrix(targets, class_predictions)
    cluster_purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return cluster_purity


def calc_external_cluster_validation_metrics(
    class_predictions: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Computer all external cluster validation metrics available in PyPOTS and return as a dictionary.

    Parameters
    ----------
    class_predictions :
        Clustering results returned by a clusterer.

    targets :
        Ground truth (correct) clustering results.

    Returns
    -------
    external_cluster_validation_metrics : dict
        A dictionary contains all external cluster validation metrics available in PyPOTS.
    """

    ri = calc_rand_index(class_predictions, targets)
    ari = calc_adjusted_rand_index(class_predictions, targets)
    nmi = calc_nmi(class_predictions, targets)
    cp = calc_cluster_purity(class_predictions, targets)

    external_cluster_validation_metrics = {
        "rand_index": ri,
        "adjusted_rand_index": ari,
        "nmi": nmi,
        "cluster_purity": cp,
    }
    return external_cluster_validation_metrics


def calc_silhouette(X: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Compute the mean Silhouette Coefficient of all samples.

    Parameters
    ----------
    X : array-like of shape (n_samples_a, n_features)
        A feature array, or learned latent representation, that can be used for clustering.

    predicted_labels : array-like of shape (n_samples)
        Predicted labels for each sample.

    Returns
    -------
    silhouette_score : float
        Mean Silhouette Coefficient for all samples. In short, the higher, the better.

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    """
    silhouette_score = metrics.silhouette_score(X, predicted_labels)
    return silhouette_score


def calc_chs(X: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Compute the Calinski and Harabasz score (also known as the Variance Ratio Criterion).

    X : array-like of shape (n_samples_a, n_features)
        A feature array, or learned latent representation, that can be used for clustering.

    predicted_labels : array-like of shape (n_samples)
        Predicted labels for each sample.

    Returns
    -------
    calinski_harabasz_score : float
        The resulting Calinski-Harabasz score. In short, the higher, the better.

    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_

    """
    calinski_harabasz_score = metrics.calinski_harabasz_score(X, predicted_labels)
    return calinski_harabasz_score


def calc_dbs(X: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Compute the Davies-Bouldin score.

    Parameters
    ----------
    X : array-like of shape (n_samples_a, n_features)
        A feature array, or learned latent representation, that can be used for clustering.

    predicted_labels : array-like of shape (n_samples)
        Predicted labels for each sample.

    Returns
    -------
    davies_bouldin_score : float
        The resulting Davies-Bouldin score. In short, the lower, the better.

    References
    ----------
    .. [1] `Davies, David L.; Bouldin, Donald W. (1979).
       "A Cluster Separation Measure"
       IEEE Transactions on Pattern Analysis and Machine Intelligence.
       PAMI-1 (2): 224-227
       <https://ieeexplore.ieee.org/document/4766909>`_

    """
    davies_bouldin_score = metrics.davies_bouldin_score(X, predicted_labels)
    return davies_bouldin_score


def calc_internal_cluster_validation_metrics(X: np.ndarray, predicted_labels: np.ndarray) -> dict:
    """Computer all internal cluster validation metrics available in PyPOTS and return as a dictionary.

    Parameters
    ----------
    X : array-like of shape (n_samples_a, n_features)
        A feature array, or learned latent representation, that can be used for clustering.

    predicted_labels : array-like of shape (n_samples)
        Predicted labels for each sample.

    Returns
    -------
    internal_cluster_validation_metrics : dict
        A dictionary contains all internal cluster validation metrics available in PyPOTS.
    """

    silhouette_score = calc_silhouette(X, predicted_labels)
    calinski_harabasz_score = calc_chs(X, predicted_labels)
    davies_bouldin_score = calc_dbs(X, predicted_labels)

    internal_cluster_validation_metrics = {
        "silhouette_score": silhouette_score,
        "calinski_harabasz_score": calinski_harabasz_score,
        "davies_bouldin_score": davies_bouldin_score,
    }
    return internal_cluster_validation_metrics
