"""
Dataset class for the imputation model PatchTST.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

from ..saits.data import DatasetForSAITS


class DatasetForPatchTST(DatasetForSAITS):
    """Actually PatchTST uses the same data strategy as SAITS, needs MIT for training."""

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
        rate: float = 0.2,
    ):
        super().__init__(data, return_X_ori, return_y, file_type, rate)
