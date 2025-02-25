"""
Dataset class for the imputation model GPT4TS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

from ..saits.data import DatasetForSAITS


class DatasetForGPT4TS(DatasetForSAITS):
    """Actually GPT4TS uses the same data strategy as SAITS, needs MIT for training, for details please refer to
    https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/9f3875dbba9f52ee534ac0902572502746f3b29d/Imputation/exp/exp_imputation.py#L99-L104

    """

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
        rate: float = 0.2,
    ):
        super().__init__(data, return_X_ori, return_y, file_type, rate)
