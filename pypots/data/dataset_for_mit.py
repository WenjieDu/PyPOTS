"""
Dataset class for self-attention models trained with MIT (masked imputation task) task.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import torch
from pycorruptor import mcar

from pypots.data.base import BaseDataset


class DatasetForMIT(BaseDataset):
    """ Dataset for models that need MIT (masked imputation task) in their training, such as SAITS.

    For more information about MIT, please refer to :cite:`du2022SAITS`.

    Parameters
    ----------
    X : tensor, shape of [n_samples, n_steps, n_features]
        Time-series feature vector.

    y : tensor, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.

    rate : float, in (0,1),
        Artificially missing rate, rate of the observed values which will be artificially masked as missing.

        Note that,
        `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
        not (number of artificially missing values) / np.product(self.data.shape),
        considering that the given data may already contain missing values,
        the latter way may be confusing because if the original missing rate >= `rate`,
        the function will do nothing, i.e. it won't play the role it has to be.

    """

    def __init__(self, X, y=None, rate=0.2):
        super().__init__(X, y)
        self.rate = rate

    def __getitem__(self, idx):
        """ Fetch data according to index.

        Parameters
        ----------
        idx : int,
            The index to fetch the specified sample.

        Returns
        -------
        dict,
            A dict contains

            index : int tensor,
                The index of the sample.

            X_intact : tensor,
                Original time-series for calculating mask imputation loss.

            X : tensor,
                Time-series data with artificially missing values for model input.

            missing_mask : tensor,
                The mask records all missing values in X.

            indicating_mask : tensor.
                The mask indicates artificially missing values in X.
        """
        X = self.X[idx]
        X_intact, X, missing_mask, indicating_mask = mcar(X, rate=self.rate)

        sample = [
            torch.tensor(idx),
            X_intact.to(torch.float32),
            X.to(torch.float32),
            missing_mask.to(torch.float32),
            indicating_mask.to(torch.float32),
        ]

        if self.y is not None:
            sample.append(
                self.y[idx].to(torch.long)
            )

        return sample
