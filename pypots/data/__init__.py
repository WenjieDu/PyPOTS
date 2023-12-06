"""
Expose all usable data manipulation classes and functions.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .base import BaseDataset
from .generating import (
    gene_complete_random_walk,
    gene_complete_random_walk_for_anomaly_detection,
    gene_complete_random_walk_for_classification,
    gene_random_walk,
    gene_physionet2012,
)
from .load_specific_datasets import (
    list_supported_datasets,
    load_specific_dataset,
)
from .saving import save_dict_into_h5
from .utils import parse_delta, sliding_window

__all__ = [
    # datasets
    "BaseDataset",
    # data generation
    "gene_complete_random_walk",
    "gene_complete_random_walk_for_anomaly_detection",
    "gene_complete_random_walk_for_classification",
    "gene_random_walk",
    "gene_physionet2012",
    # list and load datasets
    "list_supported_datasets",
    "load_specific_dataset",
    # utils
    "parse_delta",
    "sliding_window",
    # saving
    "save_dict_into_h5",
]
