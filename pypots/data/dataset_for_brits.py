"""
Dataset class for model BRITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import torch

from pypots.data.base import BaseDataset


def parse_delta(missing_mask):
    """ Generate time-gap (delta) matrix from missing masks.

    Parameters
    ----------
    missing_mask : tensor, shape of [n_samples, n_steps, n_features]
        Binary masks indicate missing values.

    Returns
    -------
    delta, array,
        Delta matrix indicates time gaps of missing values.
        Its math definition please refer to :cite:`che2018MissingData`.
    """
    # missing_mask is from X, and X's shape and type had been checked. So no need to double-check here.
    n_samples, n_steps, n_features = missing_mask.shape
    delta_collector = []
    for m_mask in missing_mask:
        delta = []
        for step in range(n_steps):
            if step == 0:
                delta.append(torch.zeros(1, n_features))
            else:
                delta.append(torch.ones(1, n_features) + (1 - m_mask[step]) * delta[-1])
        delta = torch.concat(delta, dim=0)
        delta_collector.append(delta.unsqueeze(0))
    delta = torch.concat(delta_collector, dim=0)
    return delta


class DatasetForBRITS(BaseDataset):
    """ Dataset class for BRITS.

    Parameters
    ----------
    X : tensor, shape of [n_samples, n_steps, n_features]
        Time-series data.

    y : tensor, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    """

    def __init__(self, X, y=None):
        super().__init__(X, y)

        # calculate all delta here.
        # Training will take too much time if we put delta calculation in __getitem__().
        forward_missing_mask = (~torch.isnan(X)).type(torch.float32)
        forward_X = torch.nan_to_num(X)
        forward_delta = parse_delta(forward_missing_mask)
        backward_X = torch.flip(forward_X, dims=[1])
        backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])
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
            self.data['forward']['X'][idx].to(torch.float32),
            self.data['forward']['missing_mask'][idx].to(torch.float32),
            self.data['forward']['delta'][idx].to(torch.float32),
            # for backward
            self.data['backward']['X'][idx].to(torch.float32),
            self.data['backward']['missing_mask'][idx].to(torch.float32),
            self.data['backward']['delta'][idx].to(torch.float32),
        ]

        if self.y is not None:
            sample.append(
                self.y[idx].to(torch.long)
            )

        return sample
