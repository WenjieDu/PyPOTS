"""
Dataset class for model BRITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import numpy as np
import torch

from pypots.data.base import BaseDataset


def parse_delta(missing_mask):
    """ Generate time-gap (delta) matrix from missing masks.

    Parameters
    ----------
    missing_mask : array, shape of [seq_len, n_features]
        Binary masks indicate missing values.

    Returns
    -------
    delta, array,
        Delta matrix indicates time gaps of missing values.
        Its math definition please refer to :cite:`che2018MissingData`.
    """

    assert len(missing_mask.shape) == 3, f'missing_mask should has 3 dimensions, ' \
                                         f'shape like [n_samples, seq_len, n_features], ' \
                                         f'while the input is {missing_mask.shape}'
    n_samples, seq_len, n_features = missing_mask.shape
    delta_collector = []
    for m_mask in missing_mask:
        delta = []
        for step in range(seq_len):
            if step == 0:
                delta.append(np.zeros(n_features))
            else:
                delta.append(np.ones(n_features) + (1 - m_mask[step]) * delta[-1])
        delta = np.asarray(delta)
        delta_collector.append(delta)
    return np.asarray(delta_collector)


class DatasetForBRITS(BaseDataset):
    """ Dataset class for BRITS.

    Parameters
    ----------
    X : array-like, shape of [n_samples, seq_len, n_features]
        Time-series feature vector.
    y : array-like, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    """

    def __init__(self, X, y=None):
        super(DatasetForBRITS, self).__init__(X, y)

        # calculate all delta here.
        # Training will take too much time if we put delta calculation in __getitem__().
        forward_missing_mask = (~np.isnan(X)).astype(np.float32)
        forward_X = np.nan_to_num(X)
        forward_delta = parse_delta(forward_missing_mask)
        backward_X = np.flip(forward_X, axis=1).copy()
        backward_missing_mask = np.flip(forward_missing_mask, axis=1).copy()
        backward_delta = parse_delta(backward_missing_mask)

        self.data = {
            'forward': {
                'X': forward_X,
                'missing_mask': forward_missing_mask,
                'delta': forward_delta
            },
            'backward': {
                'X': backward_X,
                'missing_mask': backward_missing_mask,
                'delta': backward_delta
            },
        }

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
            X : tensor,
                The feature vector for model input.
            missing_mask : tensor,
                The mask indicates all missing values in X.
            delta : tensor,
                The delta matrix contains time gaps of missing values.
            label (optional) : tensor,
                The target label of the time-series sample.
        """
        sample = [
            torch.tensor(idx),
            # for forward
            torch.from_numpy(self.data['forward']['X'][idx].astype('float32')),
            torch.from_numpy(self.data['forward']['missing_mask'][idx].astype('float32')),
            torch.from_numpy(self.data['forward']['delta'][idx].astype('float32')),
            # for backward
            torch.from_numpy(self.data['backward']['X'][idx].astype('float32')),
            torch.from_numpy(self.data['backward']['missing_mask'][idx].astype('float32')),
            torch.from_numpy(self.data['backward']['delta'][idx].astype('float32')),
        ]

        if self.y is not None:
            sample.append(torch.tensor(self.y[idx], dtype=torch.long))

        return sample
