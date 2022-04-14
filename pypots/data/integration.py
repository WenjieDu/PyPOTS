"""
Integrate with data functions from other libraries.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import pycorruptor as corruptor
from tsdb import AVAILABLE_DATASETS as _AVAILABLE_DATASETS
from tsdb import (
    load_specific_dataset as _load_specific_dataset,
    # delete_all_cached_data,
    # CACHED_DATASET_DIR,
    # pickle_load,
    # pickle_dump,
)

AVAILABLE_DATASETS = _AVAILABLE_DATASETS


def fill_nan_with_mask(X, mask):
    return corruptor.fill_nan_with_mask(X, mask)


def mcar(X, rate, nan=0):
    return corruptor.mcar(X, rate, nan)


def load_specific_dataset(dataset_name, use_cache=True):
    print('Loading the dataset with TSDB (https://github.com/WenjieDu/Time_Series_Database)...')
    return _load_specific_dataset(dataset_name, use_cache)
