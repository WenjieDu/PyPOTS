"""
Expose all usable data manipulation classes and functions.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .dataset import BaseDataset, SUPPORTED_DATASET_FILE_FORMATS
from .generating import gene_random_walk
from .saving import (
    save_dict_into_h5,
    load_dict_from_h5,
    pickle_dump,
    pickle_load,
)
from .utils import (
    parse_delta,
    sliding_window,
    inverse_sliding_window,
)
from .load_specific_datasets import load_specific_dataset

__all__ = [
    # base dataset classes
    "BaseDataset",
    "SUPPORTED_DATASET_FILE_FORMATS",
    # dataset generation functions
    "gene_random_walk",
    "load_specific_dataset",
    # utils
    "parse_delta",
    "sliding_window",
    "inverse_sliding_window",
    # saving
    "save_dict_into_h5",
    "load_dict_from_h5",
    "pickle_dump",
    "pickle_load",
]
