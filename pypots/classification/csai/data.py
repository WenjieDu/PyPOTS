"""

"""

# Created by Joseph Arul Raj <joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

from typing import Union
from ...imputation.csai.data import DatasetForCSAI as DatasetForCSAI_Imputation


class DatasetForCSAI(DatasetForCSAI_Imputation):
    def __init__(
        self,
        data: Union[dict, str],
        file_type: str = "hdf5",
        return_y: bool = True,
        removal_percent: float = 0.0,
        increase_factor: float = 0.1,
        compute_intervals: bool = False,
        replacement_probabilities=None,
        normalise_mean: list = [],
        normalise_std: list = [],
        training: bool = True,
    ):
        super().__init__(
            data=data,
            return_X_ori=False,
            return_y=return_y,
            file_type=file_type,
            removal_percent=removal_percent,
            increase_factor=increase_factor,
            compute_intervals=compute_intervals,
            replacement_probabilities=replacement_probabilities,
            normalise_mean=normalise_mean,
            normalise_std=normalise_std,
            training=training,
        )
