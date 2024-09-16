"""
Utilities for random data generating.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from benchpots.datasets import preprocess_physionet2012, preprocess_random_walk

from ..utils.logging import logger


def gene_random_walk(
    n_steps=24,
    n_features=10,
    n_classes=2,
    n_samples_each_class=1000,
    missing_rate=0.1,
):
    dataset_from_benchpots = preprocess_random_walk(
        n_steps,
        n_features,
        n_classes,
        n_samples_each_class,
        missing_rate,
    )
    logger.warning(
        "🚨 BenchPOTS package now is fully released and includes preprocessing functions for 170+ datasets. "
        "gene_random_walk() has been deprecated and will be removed in pypots v0.9"
    )
    logger.info(
        "🌟 Please refer to https://github.com/WenjieDu/BenchPOTS and "
        "check out the func benchpots.datasets.preprocess_physionet2012()"
    )
    return dataset_from_benchpots


def gene_physionet2012(artificially_missing_rate: float = 0.1):
    dataset_from_benchpots = preprocess_physionet2012(subset="all", rate=artificially_missing_rate)
    logger.warning(
        "🚨 BenchPOTS package now is fully released and includes preprocessing functions for 170+ datasets. "
        "gene_physionet2012() has been deprecated and will be removed in pypots v0.9"
    )
    logger.info(
        "🌟 Please refer to https://github.com/WenjieDu/BenchPOTS and "
        "check out the func benchpots.datasets.preprocess_physionet2012()"
    )
    return dataset_from_benchpots
