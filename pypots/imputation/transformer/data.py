"""
Dataset class for self-attention models trained with MIT (masked imputation task) task.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

from ..saits.data import DatasetForSAITS


class DatasetForTransformer(DatasetForSAITS):
    def __init__(
        self,
        data: Union[dict, str],
        return_labels: bool = True,
        file_type: str = "h5py",
        rate: float = 0.2,
    ):
        super().__init__(data, return_labels, file_type, rate)
