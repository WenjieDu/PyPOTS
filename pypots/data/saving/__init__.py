"""
Data saving utilities.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .h5 import save_dict_into_h5, load_dict_from_h5
from .pickle import pickle_dump, pickle_load

__all__ = [
    "save_dict_into_h5",
    "load_dict_from_h5",
    "pickle_dump",
    "pickle_load",
]
