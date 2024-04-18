"""
The package including dataset classes for PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .base import BaseDataset
from .config import SUPPORTED_DATASET_FILE_FORMATS

__all__ = [
    "BaseDataset",
    "SUPPORTED_DATASET_FILE_FORMATS",
]
