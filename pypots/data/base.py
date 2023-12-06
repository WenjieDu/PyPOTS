"""
The base class for PyPOTS datasets.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from abc import abstractmethod
from typing import Union, Optional, Tuple, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Currently we only support h5 files
SUPPORTED_DATASET_FILE_TYPE = ["h5py"]


class BaseDataset(Dataset):
    """Base dataset class in PyPOTS.

    Parameters
    ----------
    data :
        The dataset for model input, should be a dictionary including keys as 'X' and 'y',
        or a path string locating a data file.
        If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
        which is time-series data for input, can contain missing values, and y should be array-like of shape
        [n_samples], which is classification labels of X.
        If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
        key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

    return_labels :
        Whether to return labels in function __getitem__() if they exist in the given data. If `True`, for example,
        during training of classification models, the Dataset class will return labels in __getitem__() for model input.
        Otherwise, labels won't be included in the data returned by __getitem__(). This parameter exists because we
        need the defined Dataset class for all training/validating/testing stages. For those big datasets stored in h5
        files, they already have both X and y saved. But we don't read labels from the file for validating and testing
        with function _fetch_data_from_file(), which works for all three stages. Therefore, we need this parameter for
        distinction.

    file_type :
        The type of the given file if train_set and val_set are path strings.

    """

    def __init__(
        self,
        data: Union[dict, str],
        return_labels: bool = True,
        file_type: str = "h5py",
    ):
        super().__init__()
        # types and shapes had been checked after X and y input into the model
        # So they are safe to use here. No need to check again.

        self.data = data
        self.return_labels = return_labels
        if isinstance(self.data, str):  # data from file
            # check if the given file type is supported
            assert (
                file_type in SUPPORTED_DATASET_FILE_TYPE
            ), f"file_type should be one of {SUPPORTED_DATASET_FILE_TYPE}, but got {file_type}"

            self.file_type = file_type

            # open the file handle
            self.file_handle = self._open_file_handle()
            # check if X exists in the file
            assert (
                "X" in self.file_handle.keys()
            ), "The given dataset file doesn't contains X. Please double check."

        else:  # data from array
            X = data["X"]
            y = None if "y" not in data.keys() else data["y"]
            self.X, self.y = self._check_input(X, y)

        self.n_samples, self.n_steps, self.n_features = self._get_data_sizes()

        # set up function fetch_data()
        if isinstance(self.data, str):
            self.fetch_data = self._fetch_data_from_file
        else:
            self.fetch_data = self._fetch_data_from_array

    def _get_data_sizes(self) -> Tuple[int, int, int]:
        """Determine the number of samples in the dataset and return the number.

        Returns
        -------
        n_samples :
            The number of the samples in the given dataset.
        """

        if isinstance(self.data, str):
            if self.file_handle is None:
                self.file_handle = self._open_file_handle()
            n_samples = len(self.file_handle["X"])
            first_sample = self.file_handle["X"][0]
            n_steps = len(first_sample)
            n_features = first_sample.shape[-1]
        else:
            n_samples = len(self.X)
            n_steps = len(self.X[0])
            n_features = self.X[0].shape[-1]

        return n_samples, n_steps, n_features

    def __len__(self) -> int:
        return self.n_samples

    @staticmethod
    def _check_input(
        X: Union[np.ndarray, torch.Tensor, list],
        y: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
        out_dtype: str = "tensor",
    ) -> Tuple[
        Union[np.ndarray, torch.Tensor, list],
        Optional[Union[np.ndarray, torch.Tensor, list]],
    ]:
        """Check value type and shape of input X and y

        Parameters
        ----------
        X :
            Time-series data that must have a shape like [n_samples, expected_n_steps, expected_n_features].

        y :
            Labels of time-series samples (X) that must have a shape like [n_samples] or [n_samples, n_classes].

        out_dtype :
            Data type of the output, should be np.ndarray or torch.Tensor

        Returns
        -------
        X :

        y :

        """
        assert out_dtype in [
            "tensor",
            "ndarray",
        ], f'out_dtype should be "tensor" or "ndarray", but got {out_dtype}'

        is_list = isinstance(X, list)
        is_array = isinstance(X, np.ndarray)
        is_tensor = isinstance(X, torch.Tensor)
        assert is_tensor or is_array or is_list, TypeError(
            "X should be an instance of list/np.ndarray/torch.Tensor, "
            f"but got {type(X)}"
        )

        # convert the data type if in need
        if out_dtype == "tensor":
            if is_list:
                X = torch.tensor(X)
            elif is_array:
                X = torch.from_numpy(X)
            else:  # is tensor
                pass
        else:  # out_dtype is ndarray
            # convert to np.ndarray first for shape check
            if is_list:
                X = np.asarray(X)
            elif is_tensor:
                X = X.numpy()
            else:  # is ndarray
                pass

        # check the shape of X here
        X_shape = X.shape
        assert len(X_shape) == 3, (
            f"input should have 3 dimensions [n_samples, seq_len, n_features],"
            f"but got shape={X_shape}"
        )

        if y is not None:
            assert len(X) == len(y), (
                f"lengths of X and y must match, " f"but got f{len(X)} and {len(y)}"
            )
            if isinstance(y, torch.Tensor):
                y = y if out_dtype == "tensor" else y.numpy()
            elif isinstance(y, list):
                y = torch.tensor(y) if out_dtype == "tensor" else np.asarray(y)
            elif isinstance(y, np.ndarray):
                y = torch.from_numpy(y) if out_dtype == "tensor" else y
            else:
                raise TypeError(
                    "y should be an instance of list/np.ndarray/torch.Tensor, "
                    f"but got {type(y)}"
                )

        return X, y

    @abstractmethod
    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data from self.X if it is given.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            The collated data sample, a list including all necessary sample info.
        """

        X = self.X[idx].to(torch.float32)
        missing_mask = (~torch.isnan(X)).to(torch.float32)
        X = torch.nan_to_num(X)
        sample = [
            torch.tensor(idx),
            X,
            missing_mask,
        ]

        if self.y is not None and self.return_labels:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _open_file_handle(self) -> h5py.File:
        """Open the file handle for reading data from the file.

        Notes
        -----
        This function can also help confirm if the given file and file type match.

        Returns
        -------
        file_handle :

        """
        data_file_path = self.data
        try:
            file_handler = h5py.File(
                data_file_path,
                "r",
            )  # set swmr=True if the h5 file need to be written into new content during reading
        except ImportError:
            raise ImportError(
                "h5py is missing and cannot be imported. Please install it first."
            )
        except OSError as e:
            raise TypeError(
                f"{e} This probably is caused by file type error. "
                f"Please confirm that the given file {data_file_path} is an h5 file."
            )
        except Exception as e:
            raise RuntimeError(e)
        return file_handler

    @abstractmethod
    def _fetch_data_from_file(self, idx: int) -> Iterable:
        """Fetch data with the lazy-loading strategy, i.e. only loading data from the file while requesting for samples.
        Here the opened file handle doesn't load the entire dataset into RAM but only load the currently accessed slice.

        Notes
        -----
        Multi workers reading from h5 file is tricky, and I was confronted with a problem similar to
        https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/7 in 2020, please
        refer to it for more details about the problem.
        The implementation here is referred to
        https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/10
        And according to https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/37,
        pytorch v1.7.1 and h5py v3.2.0 work well, so probably updating to the latest version can avoid the
        issue I met. After all, this implementation may need to be updated in the near future.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            The collated data sample, a list including all necessary sample info.
        """

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
        missing_mask = (~torch.isnan(X)).to(torch.float32)
        X = torch.nan_to_num(X)
        sample = [
            torch.tensor(idx),
            X,
            missing_mask,
        ]

        # if the dataset has labels and is for training, then fetch it from the file
        if "y" in self.file_handle.keys() and self.return_labels:
            sample.append(self.file_handle["y"][idx].to(torch.long))

        return sample

    def __getitem__(self, idx: int) -> Iterable:
        """Fetch data according to index.

        Parameters
        ----------
        idx :
            The index to fetch the specified sample.

        Returns
        -------
        sample :
            The collated data sample, a list including all necessary sample info.
        """

        sample = self.fetch_data(idx)
        return sample
