"""
The implementation of BTTF (Bayesian Temporal Tensor Factorization) for the partially-observed time-series
forecasting task.

Refer to the paper "Chen, X., & Sun, L. (2021).
Bayesian Temporal Factorization for Multidimensional Time Series Prediction.
IEEE transactions on pattern analysis and machine intelligence."

Notes
-----
This numpy implementation is the same with the official one from https://github.com/xinychen/transdim.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import warnings
from typing import Union, Optional

import numpy as np
import torch
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from scipy.linalg import khatri_rao as kr_prod

from .modules import (
    mvnrnd_pre,
    ten2mat,
    sample_factor_u,
    sample_factor_v,
    sample_factor_x,
    sample_var_coefficient,
    ar4cast,
)
from ..base import BaseForecaster


def _BTTF(
    dense_tensor,
    sparse_tensor,
    init,
    rank,
    time_lags,
    burn_iter,
    gibbs_iter,
    multi_step=1,
):
    """Bayesian Temporal Tensor Factorization, BTTF."""

    dim1, dim2, dim3 = sparse_tensor.shape
    d = time_lags.shape[0]
    U = init["U"]
    V = init["V"]
    X = init["X"]
    if not np.isnan(sparse_tensor).any():
        ind = sparse_tensor != 0
        pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    elif np.isnan(sparse_tensor).any():
        pos_test = np.where((dense_tensor != 0) & (np.isnan(sparse_tensor)))
        ind = ~np.isnan(sparse_tensor)
        # pos_obs = np.where(ind)
        sparse_tensor[np.isnan(sparse_tensor)] = 0
    # dense_test = dense_tensor[pos_test]
    del dense_tensor
    U_plus = np.zeros((dim1, rank, gibbs_iter))
    V_plus = np.zeros((dim2, rank, gibbs_iter))
    X_plus = np.zeros((dim3 + multi_step, rank, gibbs_iter))
    A_plus = np.zeros((rank * d, rank, gibbs_iter))
    tau_plus = np.zeros(gibbs_iter)
    Sigma_plus = np.zeros((rank, rank, gibbs_iter))
    temp_hat = np.zeros(len(pos_test[0]))
    show_iter = 500
    tau = 1
    tensor_hat_plus = np.zeros(sparse_tensor.shape)
    tensor_new_plus = np.zeros((dim1, dim2, multi_step))
    for it in range(burn_iter + gibbs_iter):
        tau_ind = tau * ind
        tau_sparse_tensor = tau * sparse_tensor
        U = sample_factor_u(tau_sparse_tensor, tau_ind, U, V, X)
        V = sample_factor_v(tau_sparse_tensor, tau_ind, U, V, X)
        A, Sigma = sample_var_coefficient(X, time_lags)
        X = sample_factor_x(
            tau_sparse_tensor, tau_ind, time_lags, U, V, X, A, inv(Sigma)
        )
        tensor_hat = np.einsum("is, js, ts -> ijt", U, V, X)
        tau = np.random.gamma(
            1e-6 + 0.5 * np.sum(ind),
            1 / (1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)),
        )
        temp_hat += tensor_hat[pos_test]
        if (it + 1) % show_iter == 0 and it < burn_iter:
            # temp_hat = temp_hat / show_iter
            # logger.info('Iter: {}'.format(it + 1))
            # logger.info('MAPE: {:.6}'.format(compute_mape(dense_test, temp_hat)))
            # logger.info('RMSE: {:.6}'.format(compute_rmse(dense_test, temp_hat)))
            temp_hat = np.zeros(len(pos_test[0]))
        if it + 1 > burn_iter:
            U_plus[:, :, it - burn_iter] = U
            V_plus[:, :, it - burn_iter] = V
            A_plus[:, :, it - burn_iter] = A
            Sigma_plus[:, :, it - burn_iter] = Sigma
            tau_plus[it - burn_iter] = tau
            tensor_hat_plus += tensor_hat
            X0 = ar4cast(A, X, Sigma, time_lags, multi_step)
            X_plus[:, :, it - burn_iter] = X0
            tensor_new_plus += np.einsum("is, js, ts -> ijt", U, V, X0[-multi_step:, :])
    tensor_hat = tensor_hat_plus / gibbs_iter
    # logger.info('Imputation MAPE: {:.6}'.format(compute_mape(dense_test, tensor_hat[:, :, : dim3][pos_test])))
    # logger.info('Imputation RMSE: {:.6}'.format(compute_rmse(dense_test, tensor_hat[:, :, : dim3][pos_test])))
    tensor_hat = np.append(tensor_hat, tensor_new_plus / gibbs_iter, axis=2)
    tensor_hat[tensor_hat < 0] = 0

    return tensor_hat, U_plus, V_plus, X_plus, A_plus, Sigma_plus, tau_plus


def sample_factor_x_partial(
    tau_sparse_tensor, tau_ind, time_lags, U, V, X, A, Lambda_x, back_step
):
    """Sampling T-by-R factor matrix X."""

    dim3, rank = X.shape
    tmax = np.max(time_lags)
    tmin = np.min(time_lags)
    d = time_lags.shape[0]
    A0 = np.dstack([A] * d)
    for k in range(d):
        A0[k * rank : (k + 1) * rank, :, k] = 0
    mat0 = Lambda_x @ A.T
    mat1 = np.einsum("kij, jt -> kit", A.reshape([d, rank, rank]), Lambda_x)
    mat2 = np.einsum("kit, kjt -> ij", mat1, A.reshape([d, rank, rank]))

    var1 = kr_prod(V, U).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind[:, :, -back_step:], 2).T).reshape(
        [rank, rank, back_step]
    ) + Lambda_x[:, :, None]
    var4 = var1 @ ten2mat(tau_sparse_tensor[:, :, -back_step:], 2).T
    for t in range(dim3 - back_step, dim3):
        Mt = np.zeros((rank, rank))
        Nt = np.zeros(rank)
        Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
        index = list(range(0, d))
        if dim3 - tmax <= t < dim3 - tmin:
            index = list(np.where(t + time_lags < dim3))[0]
        if t < dim3 - tmin:
            Mt = mat2.copy()
            temp = np.zeros((rank * d, len(index)))
            n = 0
            for k in index:
                temp[:, n] = X[t + time_lags[k] - time_lags, :].reshape(rank * d)
                n += 1
            temp0 = X[t + time_lags[index], :].T - np.einsum(
                "ijk, ik -> jk", A0[:, :, index], temp
            )
            Nt = np.einsum("kij, jk -> i", mat1[index, :, :], temp0)
        var3[:, :, t + back_step - dim3] = var3[:, :, t + back_step - dim3] + Mt
        X[t, :] = mvnrnd_pre(
            solve(
                var3[:, :, t + back_step - dim3],
                var4[:, t + back_step - dim3] + Nt + Qt,
            ),
            var3[:, :, t + back_step - dim3],
        )
    return X


def _BTTF_partial(
    sparse_tensor, init, rank, time_lags, gibbs_iter, multi_step=1, gamma=10
):
    """Bayesian Temporal Tensor Factorization, BTTF."""

    dim1, dim2, dim3 = sparse_tensor.shape
    U_plus = init["U_plus"]
    V_plus = init["V_plus"]
    X_plus = init["X_plus"]
    A_plus = init["A_plus"]
    Sigma_plus = init["Sigma_plus"]
    tau_plus = init["tau_plus"]
    if not np.isnan(sparse_tensor).any():
        ind = sparse_tensor != 0
    elif np.isnan(sparse_tensor).any():
        ind = ~np.isnan(sparse_tensor)
        sparse_tensor[np.isnan(sparse_tensor)] = 0
    X_new_plus = np.zeros((dim3 + multi_step, rank, gibbs_iter))
    tensor_new_plus = np.zeros((dim1, dim2, multi_step))
    back_step = gamma * multi_step
    for it in range(gibbs_iter):
        tau_ind = tau_plus[it] * ind
        tau_sparse_tensor = tau_plus[it] * sparse_tensor
        X = sample_factor_x_partial(
            tau_sparse_tensor,
            tau_ind,
            time_lags,
            U_plus[:, :, it],
            V_plus[:, :, it],
            X_plus[:, :, it],
            A_plus[:, :, it],
            inv(Sigma_plus[:, :, it]),
            back_step,
        )
        X0 = ar4cast(A_plus[:, :, it], X, Sigma_plus[:, :, it], time_lags, multi_step)
        X_new_plus[:, :, it] = X0
        tensor_new_plus += np.einsum(
            "is, js, ts -> ijt", U_plus[:, :, it], V_plus[:, :, it], X0[-multi_step:, :]
        )
    tensor_hat = tensor_new_plus / gibbs_iter
    tensor_hat[tensor_hat < 0] = 0

    return tensor_hat, U_plus, V_plus, X_new_plus, A_plus, Sigma_plus, tau_plus


def BTTF_forecast(
    dense_tensor,
    sparse_tensor,
    pred_step,
    multi_step,
    rank,
    time_lags,
    burn_iter,
    gibbs_iter,
    gamma=10,
):
    dim1, dim2, T = dense_tensor.shape
    start_time = T - pred_step
    max_count = int(np.ceil(pred_step / multi_step))
    tensor_hat = np.zeros((dim1, dim2, max_count * multi_step))

    # t==0
    init = {
        "U": 0.1 * np.random.randn(dim1, rank),
        "V": 0.1 * np.random.randn(dim2, rank),
        "X": 0.1 * np.random.randn(start_time, rank),
    }
    tensor, U, V, X_new, A, Sigma, tau = _BTTF(
        dense_tensor[:, :, :start_time],
        sparse_tensor[:, :, :start_time],
        init,
        rank,
        time_lags,
        burn_iter,
        gibbs_iter,
        multi_step,
    )
    tensor_hat[:, :, 0:multi_step] = tensor[:, :, -multi_step:]
    # 1<= t <max_count
    for t in range(1, max_count):
        init = {
            "U_plus": U,
            "V_plus": V,
            "X_plus": X_new,
            "A_plus": A,
            "Sigma_plus": Sigma,
            "tau_plus": tau,
        }
        tensor, U, V, X_new, A, Sigma, tau = _BTTF_partial(
            sparse_tensor[:, :, : start_time + t * multi_step],
            init,
            rank,
            time_lags,
            gibbs_iter,
            multi_step,
            gamma,
        )
        tensor_hat[:, :, t * multi_step : (t + 1) * multi_step] = tensor[
            :, :, -multi_step:
        ]
    return tensor_hat


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
        file_type="h5py",
    ) -> None:
        """Train the forecaster on the given data.

        Warnings
        --------
        BTTF does not need to run fit().
        Please run func ``forecast()`` directly.

        """
        warnings.warn("Please run func forecast(X) directly.")

    def forecast(
        self,
        X: Union[dict, str],
        file_type: str = "h5py",
    ) -> np.ndarray:
        assert not isinstance(
            X, str
        ), "BTTF so far does not accept file input. It needs a specified Dataset class."

        X = X["X"]
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
        pred = pred.transpose((0, 2, 1))
        return pred
