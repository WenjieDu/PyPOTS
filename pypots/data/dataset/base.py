"""
The base dataset class.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional, Tuple, Iterable

import h5py
import numpy as np
import torch
from numpy import ndarray
from pygrinder import fill_and_get_mask_torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import SUPPORTED_DATASET_FILE_FORMATS
from ..saving import load_dict_from_h5
from ..utils import turn_data_into_specified_dtype


class BaseDataset(Dataset):
    """Base dataset class for models in PyPOTS.

    Parameters
    ----------
    data :
        The dataset for model input, should be a dictionary or
        a path string locating a data file that is in supported formats.
        If it is a dict, 'X' is mandatory and 'X_ori', 'X_pred', and 'y' are optional.
        ``X`` is time-series data for input and could contain missing values.
        It should be array-like of shape [n_samples, n_steps (sequence length), n_features].
        ``X_ori`` is optional. If ``X`` is constructed from ``X_ori`` with specially designed artificial missingness,
        your model may need ``X_ori`` for evaluation or loss calculation during training (e.g. SAITS).
        It should have the same shape as ``X``.
        ``X_pred`` is optional, and it is the forecasting results for the model to predict in forecasting tasks.
        ``X_pred`` should be array-like of shape [n_samples, n_steps (sequence length), n_features], and its shape
        could different from ``X``. But remember that ``X_pred`` contains time series forecasting results of ``X``,
        hence it has the same number of samples as ``X``, i.e. n_samples of them are the same, but their n_steps
        and n_features could be different. ``X_pred`` could have missing values as well as ``X``.
        ``y`` should be array-like of shape [n_samples], which is classification labels of X.
        If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
        key-value pairs like a dict, and it has to include keys as 'X', etc.

    return_X_ori :
        Whether to return X_ori and indicating_mask in function __getitem__() if it is given. If `True`, for example,
        during training of models that need the original X, the Dataset class will return X_ori in __getitem__() for
        model input. Otherwise, X_ori and indicating mask won't be included in the data list returned by __getitem__().

    return_X_pred :
        Whether to return X_pred and X_pred_missing_mask in function __getitem__() if it is given.
        If `True`, for example, during training of forecasting models, the Dataset class will return forecasting X
        in __getitem__() for model input. Otherwise, X_pred and its missing mask X_pred_missing_mask won't be included
        in the data list returned by __getitem__().

    return_y :
        Whether to return y (i.e. labels) in function __getitem__() if they exist in the given data. If `True`,
        for example, during training of classification models, the Dataset class will return labels in __getitem__()
        for model input. Otherwise, labels won't be included in the data returned by __getitem__().
        This parameter exists because we need the defined Dataset class for all training/validating/testing stages.
        For those big datasets stored in h5 files, they already have both X and y saved.
        But we don't read labels from the file for validating and testing with function _fetch_data_from_file(),
        which works for all three stages. Therefore, we need this parameter for distinction.

    file_type :
        The type of the given file if train_set and val_set are path strings.

    """

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_X_pred: bool,
        return_y: bool,
        file_type: str = "hdf5",
    ):
        super().__init__()
        # types and shapes had been checked after X and y input into the model
        # So they are safe to use here. No need to check again.

        self.data = data
        self.return_X_ori = return_X_ori
        self.return_X_pred = return_X_pred
        self.return_y = return_y
        self.file_type = file_type

        # initialize the following attributes
        self.X = None
        self.X_ori = None
        self.missing_mask = None
        self.indicating_mask = None
        self.X_pred = None
        self.X_pred_missing_mask = None
        self.y = None
        self.file_handle = None
        self.fetch_data = None
        self.n_samples: int = 0  # num of the samples in the dataset
        self.n_steps: int = 0  # num of the time steps in each sample
        self.n_features: int = 0  # num of the features in each sample
        self.n_pred_steps: int = 0  # num of the time steps in each forecasting sample
        self.n_pred_features: int = 0  # num of the features in each forecasting sample

        # check the data type and set up the fetch_data function
        if isinstance(self.data, str):  # data from file
            # check if the given file type is supported
            assert (
                file_type in SUPPORTED_DATASET_FILE_FORMATS
            ), f"file_type should be one of {SUPPORTED_DATASET_FILE_FORMATS}, but got {file_type}"
            self.file_type = file_type

            # open the file handle
            self.file_handle = self._open_file_handle()
            # check if X exists in the file
            assert "X" in self.file_handle.keys(), "The given dataset file doesn't contains X. Please double check."
            # check whether X_ori, X_pred, and y exist in the file if they are required
            if self.return_X_ori:
                assert (
                    "X_ori" in self.file_handle.keys()
                ), "The given dataset file doesn't contains X_ori. Please double check."
            if self.return_X_pred:
                assert (
                    "X_pred" in self.file_handle.keys()
                ), "The given dataset file doesn't contains X_pred. Please double check."
            if self.return_y:
                assert "y" in self.file_handle.keys(), "The given dataset file doesn't contains y. Please double check."

            # set up the function fetch_data() to fetch data from file
            self.fetch_data = self._fetch_data_from_file

        else:  # data from array
            # check if X exists in the dictionary
            assert "X" in self.data.keys(), "The given dataset dictionary doesn't contains X. Please double check."
            # check whether X_ori, X_pred, and y exist in the file if they are required
            if self.return_X_ori:
                assert (
                    "X_ori" in self.data.keys()
                ), "The given dataset dictionary doesn't contains X_ori. Please double check."
            if self.return_X_pred:
                assert (
                    "X_pred" in self.data.keys()
                ), "The given dataset dictionary doesn't contains X_pred. Please double check."
            if self.return_y:
                assert "y" in self.data.keys(), "The given dataset dictionary doesn't contains y. Please double check."

            X = data["X"]
            X_ori = None if "X_ori" not in data.keys() else data["X_ori"]
            X_pred = None if "X_pred" not in data.keys() else data["X_pred"]
            y = None if "y" not in data.keys() else data["y"]
            self.X, self.X_ori, self.X_pred, self.y = self._check_array_input(X, X_ori, X_pred, y, "tensor")

            if self.return_X_ori:
                # Only when X_ori is given and fixed, we fill the missing values in X here in advance.
                # Otherwise, we may need original X with missing values to generate X_ori, e.g. in DatasetForSAITS.
                self.X, self.missing_mask = fill_and_get_mask_torch(self.X)

                self.X_ori, X_ori_missing_mask = fill_and_get_mask_torch(self.X_ori)
                indicating_mask = X_ori_missing_mask - self.missing_mask
                self.indicating_mask = indicating_mask.to(torch.float32)

            if self.return_X_pred:
                self.X_pred, self.X_pred_missing_mask = fill_and_get_mask_torch(self.X_pred)

            # set up the function fetch_data() to fetch data from array
            self.fetch_data = self._fetch_data_from_array

        # get the sizes of the dataset
        (
            self.n_samples,
            self.n_steps,
            self.n_features,
            self.n_pred_steps,
            self.n_pred_features,
        ) = self._get_data_sizes()

    def _get_data_sizes(self) -> Tuple[int, ...]:
        """Detect the data sample sizes in the dataset and return the numbers.

        Returns
        -------
        n_samples :
            The number of the samples in the given dataset.

        n_steps :
            The number of each sample's time steps in the given dataset.

        n_features :
            The number of each sample's features in the given dataset.

        n_pred_steps :
            The number of each sample's forecasting time steps in the given dataset.
            Return as 0 if the dataset does not contain X_pred which includes data samples for forecasting tasks.

        n_pred_features :
            The number of each sample's forecasting features in the given dataset.
            Return as 0 if the dataset does not contain X_pred which includes data samples for forecasting tasks.
        """

        # initialize the sizes
        n_samples, n_steps, n_features, n_pred_steps, n_pred_features = 0, 0, 0, 0, 0

        if isinstance(self.data, str):
            if self.file_handle is None:
                self.file_handle = self._open_file_handle()
            n_samples = len(self.file_handle["X"])
            first_sample = self.file_handle["X"][0]
            n_steps = len(first_sample)
            n_features = first_sample.shape[-1]

            if self.return_X_pred:
                first_pred_sample = self.file_handle["X_pred"][0]
                n_pred_steps = len(first_pred_sample)
                n_pred_features = first_pred_sample.shape[-1]
        else:
            n_samples = len(self.X)
            n_steps = len(self.X[0])
            n_features = self.X[0].shape[-1]

            if self.return_X_pred:
                n_pred_steps = len(self.X_pred[0])
                n_pred_features = self.X_pred[0].shape[-1]

        return n_samples, n_steps, n_features, n_pred_steps, n_pred_features

    def __len__(self) -> int:
        return self.n_samples

    @staticmethod
    def _check_array_input(
        X: Union[np.ndarray, torch.Tensor],
        X_ori: Optional[Union[np.ndarray, torch.Tensor]] = None,
        X_pred: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        out_dtype: str = "tensor",
    ) -> Tuple[
        Union[Tensor, ndarray],
        Optional[Union[Tensor, ndarray]],
        Optional[Union[Tensor, ndarray]],
        Optional[Union[Tensor, ndarray]],
    ]:
        """Check value type and shape of input X and y

        Parameters
        ----------
        X :
            The data samples for testing, should be array-like with shape [n_samples, n_steps, n_features], or a path
            string locating a data file, e.g. h5 file.

        X_ori :
            If X is with artificial missingness, X_ori is the original X without artificial missing values.
            It must have the same shape as X. If X_ori is with original missing values, should be left as NaN.

        X_pred :
            The forecasting results of X, should be array-like with shape [n_samples, n_pred_steps, n_features],
            or a path string locating a data file, e.g. h5 file.

        y :
            Labels of time-series samples (X) that must have a shape like [n_samples] or [n_samples, n_classes].

        out_dtype :
            Data type of the output, should be np.ndarray or torch.Tensor

        Returns
        -------
        X :

        X_ori :

        X_pred :

        y :

        """
        assert out_dtype in [
            "tensor",
            "ndarray",
        ], f'out_dtype should be "tensor" or "ndarray", but got {out_dtype}'

        # change the data type of X
        X = turn_data_into_specified_dtype(X, out_dtype)
        X = X.to(torch.float32) if out_dtype == "tensor" else X

        # check the shape of X here
        X_shape = X.shape
        assert len(X_shape) == 3, (
            f"input should have 3 dimensions [n_samples, seq_len, n_features]," f"but got X: {X_shape}"
        )
        if X_ori is not None:
            X_ori = turn_data_into_specified_dtype(X_ori, out_dtype)
            X_ori = X_ori.to(torch.float32) if out_dtype == "tensor" else X_ori
            assert (
                X_shape == X_ori.shape
            ), f"X and X_ori must have matched shape, but got X: f{X.shape} and X_ori: {X_ori.shape}"

        if X_pred is not None:
            X_pred = turn_data_into_specified_dtype(X_pred, out_dtype)
            X_pred = X_pred.to(torch.float32) if out_dtype == "tensor" else X_pred
            assert len(X) == len(
                X_pred
            ), f"X and X_pred must have the same number of samples, but got X: f{X.shape} and X_pred: {X_pred.shape}"

        if y is not None:
            assert len(X) == len(y), f"lengths of X and y must match, " f"but got f{len(X)} and {len(y)}"
            y = turn_data_into_specified_dtype(y, out_dtype)
            y = y.to(torch.long) if out_dtype == "tensor" else y

        return X, X_ori, X_pred, y

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

        X = self.X[idx]

        if self.return_X_ori:
            # if X_ori is given, fetch missing mask from self.missing_mask that has been created in __init__()
            missing_mask = self.missing_mask[idx]
            X_ori = self.X_ori[idx]
            indicating_mask = self.indicating_mask[idx]
            sample = [torch.tensor(idx), X, missing_mask, X_ori, indicating_mask]
        else:
            X, missing_mask = fill_and_get_mask_torch(X)
            sample = [torch.tensor(idx), X, missing_mask]

        if self.return_X_pred:
            X_pred = self.X_pred[idx]
            X_pred_missing_mask = self.X_pred_missing_mask[idx]
            sample.extend([X_pred, X_pred_missing_mask])

        if self.return_y:
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
            raise ImportError("h5py is missing and cannot be imported. Please install it first.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{e}")
        except OSError as e:
            raise TypeError(
                f"{e}\n"
                f"Check out the above error log. This probably is caused by file type error. "
                f"Please confirm that the given file {data_file_path} is an h5 file."
            )
        except Exception as e:
            raise RuntimeError(e)
        return file_handler

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
        X, missing_mask = fill_and_get_mask_torch(X)
        sample = [
            torch.tensor(idx),
            X,
            missing_mask,
        ]

        if self.return_X_ori:
            X_ori = torch.from_numpy(self.file_handle["X_ori"][idx]).to(torch.float32)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)
            sample.extend([X_ori, indicating_mask])

        if self.return_X_pred:
            X_pred = torch.from_numpy(self.file_handle["X_pred"][idx]).to(torch.float32)
            X_pred, X_pred_missing_mask = fill_and_get_mask_torch(X_pred)
            sample.extend([X_pred, X_pred_missing_mask])

        # if the dataset has labels and is for training, then fetch it from the file
        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample

    def fetch_entire_dataset(self) -> dict:
        """Fetch the entire dataset from the given data source.

        Returns
        -------
        data :
            The entire dataset in a dictionary fetched from the given data source.

        """
        if isinstance(self.data, str):  # data from file
            data = load_dict_from_h5(self.data)
        else:
            data = self.data
        return data

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
