"""
The core wrapper assembles the submodules of BTTF forecasting model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from scipy.linalg import khatri_rao as kr_prod

from .submodules import (
    mvnrnd_pre,
    ten2mat,
    sample_factor_u,
    sample_factor_v,
    sample_factor_x,
    sample_var_coefficient,
    ar4cast,
)


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
        X = sample_factor_x(tau_sparse_tensor, tau_ind, time_lags, U, V, X, A, inv(Sigma))
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


def sample_factor_x_partial(tau_sparse_tensor, tau_ind, time_lags, U, V, X, A, Lambda_x, back_step):
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
    var3 = (var2 @ ten2mat(tau_ind[:, :, -back_step:], 2).T).reshape([rank, rank, back_step]) + Lambda_x[:, :, None]
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
            temp0 = X[t + time_lags[index], :].T - np.einsum("ijk, ik -> jk", A0[:, :, index], temp)
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


def _BTTF_partial(sparse_tensor, init, rank, time_lags, gibbs_iter, multi_step=1, gamma=10):
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
        tensor_new_plus += np.einsum("is, js, ts -> ijt", U_plus[:, :, it], V_plus[:, :, it], X0[-multi_step:, :])
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
    assert start_time > -1, (
        "start_time should be larger than -1, "
        "namely the number of the input tensor's time steps should be larger than pred_step."
    )
    assert start_time >= np.max(time_lags), f"start_time {start_time} should be >= max(time_lags) {np.max(time_lags)}"
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
        tensor_hat[:, :, t * multi_step : (t + 1) * multi_step] = tensor[:, :, -multi_step:]
    return tensor_hat
