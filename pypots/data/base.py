"""
Utilities for data manipulation
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from abc import abstractmethod
import h5py
import torch
from torch.utils.data import Dataset

# Currently we only support h5 files
SUPPORTED_DATASET_FILE_TYPE = ["h5py"]


class BaseDataset(Dataset):
    """Base dataset class in PyPOTS.

    Parameters
    ----------
    X : tensor, shape of [n_samples, n_steps, n_features]
        Time-series feature vector.

    y : tensor, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.

    file_path : str,
        The path to the dataset file.

    file_type : str,
        The type of the given file, should be one of `numpy`, `h5py`, `pickle`.
    """

    def __init__(self, X=None, y=None, file_path=None, file_type="h5py"):
        super().__init__()
        # types and shapes had been checked after X and y input into the model
        # So they are safe to use here. No need to check again.

        assert X is None and file_path is None, f"X and file_path cannot both be None."
        assert (
            X is not None and file_path is not None
        ), f"X and file_path cannot both be given. Either of them should be given."
        assert (
            file_type in SUPPORTED_DATASET_FILE_TYPE
        ), f"file_type should be one of {SUPPORTED_DATASET_FILE_TYPE}, but got {file_type}"

        self.X = X
        self.y = y
        self.file_path = file_path
        self.file_type = file_type
        if self.file_path is not None:
            self.file_handler = self._open_file_handle()
            assert (
                "X" in self.file_handler.keys()
            ), "The given dataset file doesn't contains X. Please double check."
        self.sample_num = self._get_sample_num()

        if self.X is not None:
            self.fetch_data = self._fetch_data_from_array
        else:
            self.fetch_data = self._fetch_data_from_file

    def _get_sample_num(self):
        """Determine the number of samples in the dataset and return the number.

        Returns
        -------
        sample_num : int
            The number of the samples in the given dataset.
        """
        if self.X is not None:
            sample_num = len(self.X)
        elif self.file_type == "h5py":
            with h5py.File(self.file_path, "r") as hf:
                sample_num = len(hf["X"])
        else:
            raise TypeError(f"So far only h5py is supported.")

        return sample_num

    def __len__(self):
        return self.sample_num

    @abstractmethod
    def _fetch_data_from_array(self, idx):
        """Fetch data from self.X if it is given.

        Parameters
        ----------
        idx : int,
            The index of the sample to be return.

        Returns
        -------
        sample : list,
            The collated data sample, a list including all necessary sample info.
        """

        X = self.X[idx]
        missing_mask = ~torch.isnan(X)
        X = torch.nan_to_num(X)
        sample = [
            torch.tensor(idx),
            X.to(torch.float32),
            missing_mask.to(torch.float32),
        ]

        if self.y is not None:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _open_file_handle(self):
        """Open the file handle for reading data from the file.

        Notes
        -----
        This function can also help confirm if the given file and file type match.

        Returns
        -------
        file_handle : file.

        """
        try:
            file_handler = h5py.File(
                self.file_path, "r"
            )  # set swmr=True if the h5 file need to be written into new content during reading
        except OSError as e:
            raise TypeError(
                f"{e} This probably is caused by file type error. "
                f"Please confirm that the given file {self.file_path} is an h5 file."
            )
        except Exception as e:
            raise RuntimeError(e)
        return file_handler

    @abstractmethod
    def _fetch_data_from_file(self, idx):
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
        idx : int,
            The index of the sample to be return.

        Returns
        -------
        sample : list,
            The collated data sample, a list including all necessary sample info.
        """

        if self.file_handler is None:
            self.file_handler = self._open_file_handle()

        X = self.file_handler["X"][idx]
        missing_mask = ~torch.isnan(X)
        X = torch.nan_to_num(X)
        sample = [
            torch.tensor(idx),
            X.to(torch.float32),
            missing_mask.to(torch.float32),
        ]

        if (
            "y" in self.file_handler.keys()
        ):  # if the dataset has labels, then fetch it from the file
            sample.append(self.file_handler["y"][idx].to(torch.long))

        return sample

    def __getitem__(self, idx):
        """Fetch data according to index.

        Parameters
        ----------
        idx : int,
            The index to fetch the specified sample.

        Returns
        -------
        sample : list,
            The collated data sample, a list including all necessary sample info.
        """

        sample = self.fetch_data(idx)
        return sample
