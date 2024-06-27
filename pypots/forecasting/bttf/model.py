"""
The implementation of BTTF (Bayesian Temporal Tensor Factorization) for the partially-observed time-series
forecasting task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import warnings
from typing import Union, Optional

import numpy as np
import torch

from .core import BTTF_forecast
from ..base import BaseForecaster


class BTTF(BaseForecaster):
    """The implementation of the BTTF model :cite:`chen2021BTMF`.

    Parameters
    ----------
    n_steps : int,
        The number of time steps in the time-series data sample.

    n_features : int,
        The number of features in the time-series data sample.

    pred_step : int,
        The number of time steps to forecast.

    rank : int,
        The rank of the low-rank tensor.

    time_lags : list,
        The time lags.

    burn_iter : int,
        The number of burn-in iterations.

    gibbs_iter : int,
        The number of Gibbs iterations.

    multi_step : int, default = 1,
        The number of time steps to forecast at each iteration.

    device :
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    Notes
    -----
    1). ``n_steps`` must be larger than ``pred_step``;

    2). ``n_steps - pred_step`` must be larger than ``max(time_lags)``;

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        pred_step: int,
        rank: int,
        time_lags: list,
        burn_iter: int,
        gibbs_iter: int,
        multi_step: int = 1,
        device: Optional[Union[str, torch.device, list]] = None,
    ):
        super().__init__(device)
        self.n_steps = n_steps
        self.n_features = n_features
        self.pred_step = pred_step
        self.multi_step = multi_step
        self.rank = rank
        self.time_lags = np.asarray(time_lags)
        self.burn_iter = burn_iter
        self.gibbs_iter = gibbs_iter

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the forecaster on the given data.

        Warnings
        --------
        BTTF does not need to run fit().
        Please run func ``forecast()`` directly.

        """
        warnings.warn("Please run func forecast(X) directly.")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        assert not isinstance(
            test_set, str
        ), "BTTF so far does not accept file input. It needs a specified Dataset class."

        X = test_set["X"]
        X = X.transpose((0, 2, 1))

        pred = BTTF_forecast(
            X,
            X.copy(),
            self.pred_step,
            self.multi_step,
            self.rank,
            self.time_lags,
            self.burn_iter,
            self.gibbs_iter,
        )
        forecasting = pred.transpose((0, 2, 1))
        result_dict = {
            "forecasting": forecasting,
        }
        return result_dict

    def forecast(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Forecast the future of the input with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, n_pred_steps, n_features],
            Forecasting results.
        """

        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["forecasting"]
