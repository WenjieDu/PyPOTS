"""
Integrate with data functions from other libraries.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import numpy as np
import pycorruptor as corruptor
from tsdb import (
    load_dataset as _load_dataset,
    CACHED_DATASET_DIR as _CACHED_DATASET_DIR,
    list_available_datasets as _list_available_datasets,
    list_database as _list_database,
    list_cached_data as _list_cached_data,
    delete_cached_data as _delete_cached_data,
    pickle_load as _pickle_load,
    pickle_dump as _pickle_dump,
)

CACHED_DATASET_DIR = _CACHED_DATASET_DIR
list_database = _list_database
list_cached_data = _list_cached_data
delete_cached_data = _delete_cached_data
pickle_load = _pickle_load
pickle_dump = _pickle_dump


def cal_missing_rate(X):
    return corruptor.cal_missing_rate(X)


def masked_fill(X, mask, val=np.nan):
    return corruptor.masked_fill(X, mask, val)


def mcar(X, rate, nan=0):
    return corruptor.mcar(X, rate, nan)


def load_specific_dataset(dataset_name, use_cache=True):
    print(f'Loading the dataset {dataset_name} with TSDB (https://github.com/WenjieDu/Time_Series_Database)...')
    return _load_dataset(dataset_name, use_cache)


def list_available_datasets():
    print('Obtaining the list of available datasets in TSDB (https://github.com/WenjieDu/Time_Series_Database)...')
    return _list_available_datasets()
