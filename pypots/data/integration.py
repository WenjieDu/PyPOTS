"""
Integrate with data functions from other libraries.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import pycorruptor as corruptor
from tsdb import (
    pickle_load as _pickle_load,
    pickle_dump as _pickle_dump,
)

pickle_load = _pickle_load
pickle_dump = _pickle_dump


def cal_missing_rate(X):
    return corruptor.cal_missing_rate(X)


def masked_fill(X, mask, val):
    return corruptor.masked_fill(X, mask, val)


def mcar(X, rate, nan=0):
    return corruptor.mcar(X, rate, nan)
