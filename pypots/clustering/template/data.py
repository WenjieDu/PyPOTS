"""
Dataset class for the clustering model YourNewModel.

TODO: modify the above description for your model's dataset class.

"""

# Created by Your Name <Your contact email> TODO: modify the author information.
# License: BSD-3-Clause

from typing import Union, Iterable

from ...data.dataset.base import BaseDataset


# TODO: define your new dataset class here. Remove or add arguments as needed.
class DatasetForYourNewModel(BaseDataset):
    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_X_pred: bool,
        return_y: bool,
        file_type: str = "hdf5",
    ):
        super().__init__(
            data=data,
            return_X_ori=return_X_ori,
            return_X_pred=return_X_pred,
            return_y=return_y,
            file_type=file_type,
        )

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        raise NotImplementedError

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        raise NotImplementedError
