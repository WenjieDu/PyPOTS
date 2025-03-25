"""
Dataset class for the imputation model TRMF.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

from ...data.dataset.base import BaseDataset


class DatasetForTRMF(BaseDataset):
    def __init__(
        self,
        data: Union[dict, str],
    ):
        super().__init__(
            data,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
        )
