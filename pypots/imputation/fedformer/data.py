"""
Dataset class for FEDformer.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

from ..saits.data import DatasetForSAITS


class DatasetForFEDformer(DatasetForSAITS):
    """Actually FEDformer uses the same data strategy as SAITS, needs MIT for training."""

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_labels: bool,
        file_type: str = "h5py",
        rate: float = 0.2,
    ):
        super().__init__(data, return_X_ori, return_labels, file_type, rate)
