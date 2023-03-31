"""
Implementation of BTTF: Bayesian Temporal Tensor Factorization.
This numpy implementation is the same with the official one from https://github.com/xinychen/transdim.
Refer to :cite:`chen2021BTMF`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import warnings

import numpy as np
from numpy.linalg import cholesky as cholesky_lower
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from numpy.random import multivariate_normal as mvnrnd
from numpy.random import normal as normrnd
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import khatri_rao as kr_prod
from scipy.linalg import solve_triangular as solve_ut
from scipy.stats import invwishart
from scipy.stats import wishart

from pypots.forecasting.base import BaseForecaster
from pypots.utils.logging import logger


def mvnrnd_pre(mu, Lambda):
    src = normrnd(size=(mu.shape[0],))
    return (
        solve_ut(
            cholesky_upper(Lambda, overwrite_a=True, check_finite=False),
            src,
            lower=False,
            check_finite=False,
            overwrite_b=True,
        )
        + mu
    )


def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order="F")


def sample_factor_u(tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
    """Sampling M-by-R factor matrix U and its hyper-parameters (mu_u, Lambda_u)."""

    dim1, rank = U.shape
    U_bar = np.mean(U, axis=0)
    temp = dim1 / (dim1 + beta0)
    var_mu_hyper = temp * U_bar
    var_U_hyper = inv(
        np.eye(rank) + cov_mat(U, U_bar) + temp * beta0 * np.outer(U_bar, U_bar)
    )
    var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_U_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim1 + beta0) * var_Lambda_hyper)

    var1 = kr_prod(X, V).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, 0).T).reshape(
        [rank, rank, dim1]
    ) + var_Lambda_hyper[:, :, None]
    var4 = (
        var1 @ ten2mat(tau_sparse_tensor, 0).T
        + (var_Lambda_hyper @ var_mu_hyper)[:, None]
    )
    for i in range(dim1):
        U[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])

    return U


def sample_factor_v(tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
    """Sampling N-by-R factor matrix V and its hyper-parameters (mu_v, Lambda_v)."""

    dim2, rank = V.shape
    V_bar = np.mean(V, axis=0)
    temp = dim2 / (dim2 + beta0)
    var_mu_hyper = temp * V_bar
    var_V_hyper = inv(
        np.eye(rank) + cov_mat(V, V_bar) + temp * beta0 * np.outer(V_bar, V_bar)
    )
    var_Lambda_hyper = wishart.rvs(df=dim2 + rank, scale=var_V_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim2 + beta0) * var_Lambda_hyper)

    var1 = kr_prod(X, U).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, 1).T).reshape(
        [rank, rank, dim2]
    ) + var_Lambda_hyper[:, :, None]
    var4 = (
        var1 @ ten2mat(tau_sparse_tensor, 1).T
        + (var_Lambda_hyper @ var_mu_hyper)[:, None]
    )
    for j in range(dim2):
        V[j, :] = mvnrnd_pre(solve(var3[:, :, j], var4[:, j]), var3[:, :, j])

    return V


def mnrnd(M, U, V):
    """
    Generate matrix normal distributed random matrix.
    M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
    """
    dim1, dim2 = M.shape
    X0 = np.random.randn(dim1, dim2)
    P = cholesky_lower(U)
    Q = cholesky_lower(V)

    return M + P @ X0 @ Q.T


def sample_var_coefficient(X, time_lags):
    dim, rank = X.shape
    d = time_lags.shape[0]
    tmax = np.max(time_lags)

    Z_mat = X[tmax:dim, :]
    Q_mat = np.zeros((dim - tmax, rank * d))
    for k in range(d):
        Q_mat[:, k * rank : (k + 1) * rank] = X[
            tmax - time_lags[k] : dim - time_lags[k], :
        ]
    var_Psi0 = np.eye(rank * d) + Q_mat.T @ Q_mat
    var_Psi = inv(var_Psi0)
    var_M = var_Psi @ Q_mat.T @ Z_mat
    var_S = np.eye(rank) + Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M
    Sigma = invwishart.rvs(df=rank + dim - tmax, scale=var_S)

    return mnrnd(var_M, var_Psi, Sigma), Sigma


def sample_factor_x(tau_sparse_tensor, tau_ind, time_lags, U, V, X, A, Lambda_x):
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
    var3 = (var2 @ ten2mat(tau_ind, 2).T).reshape([rank, rank, dim3]) + Lambda_x[
        :, :, None
    ]
    var4 = var1 @ ten2mat(tau_sparse_tensor, 2).T
    for t in range(dim3):
        Mt = np.zeros((rank, rank))
        Nt = np.zeros(rank)
        Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
        index = list(range(0, d))
        if dim3 - tmax <= t < dim3 - tmin:
            index = list(np.where(t + time_lags < dim3))[0]
        elif t < tmax:
            Qt = np.zeros(rank)
            index = list(np.where(t + time_lags >= tmax))[0]
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

        var3[:, :, t] = var3[:, :, t] + Mt
        if t < tmax:
            var3[:, :, t] = var3[:, :, t] - Lambda_x + np.eye(rank)
        X[t, :] = mvnrnd_pre(solve(var3[:, :, t], var4[:, t] + Nt + Qt), var3[:, :, t])

    return X


def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]


def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])


def ar4cast(A, X, Sigma, time_lags, multi_step):
    dim, rank = X.shape
    d = time_lags.shape[0]
    X_new = np.append(X, np.zeros((multi_step, rank)), axis=0)
    for t in range(multi_step):
        var = A.T @ X_new[dim + t - time_lags, :].reshape(rank * d)
        X_new[dim + t, :] = mvnrnd(var, Sigma)
    return X_new


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
    for t in range(max_count):
        if t == 0:
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
        else:
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
    def __init__(
        self,
        n_steps,
        n_features,
        pred_step,
        multi_step,
        rank,
        time_lags,
        burn_iter,
        gibbs_iter,
        device=None,
    ):
        super().__init__(device)
        self.n_steps = n_steps
        self.n_features = n_features
        self.pred_step = pred_step
        self.multi_step = multi_step
        self.rank = rank
        self.time_lags = time_lags
        self.burn_iter = burn_iter
        self.gibbs_iter = gibbs_iter

    def fit(self, train_set, val_set=None, file_type="h5py"):
        warnings.warn("Please run func forecast(X) directly.")

    def forecast(self, X, file_type="h5py"):
        """Forecast the future the input with the trained model.

        Parameters
        ----------
        X : array-like or str,
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type : str, default = "h5py"
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, prediction_horizon, n_features],
            Forecasting results.
        """
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
