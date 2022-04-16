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
    X : array-like, shape of [n_samples, seq_len, n_features]
        Time-series feature vector.

    y : array-like, shape of [n_samples], optional, default=None,
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
        super(DatasetForMIT, self).__init__(X, y)
        self.rate = rate

    def __getitem__(self, idx):
        X = self.X[idx]
        X_intact, X, missing_mask, indicating_mask = mcar(X, rate=self.rate)

        sample = [
            torch.tensor(idx),
            torch.from_numpy(X_intact.astype('float32')),
            torch.from_numpy(X.astype('float32')),
            torch.from_numpy(missing_mask.astype('float32')),
            torch.from_numpy(indicating_mask.astype('float32')),
        ]

        if self.y is not None:
            sample.append(torch.tensor(self.y[idx], dtype=torch.long))

        return sample
