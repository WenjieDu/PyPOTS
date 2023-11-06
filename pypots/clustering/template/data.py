"""
Dataset class for YourNewModel.

TODO: modify the above description with your model's information.

"""

# Created by Your Name <Your contact email> TODO: modify the author information.
# License: BSD-3-Clause

from typing import Union, Iterable

from ...data.base import BaseDataset


class DatasetForYourNewModel(BaseDataset):
    def __init__(
        self,
        data: Union[dict, str],
        return_labels: bool = True,
        file_type: str = "h5py",
    ):
        super().__init__(data, return_labels, file_type)

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        raise NotImplementedError

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        raise NotImplementedError
