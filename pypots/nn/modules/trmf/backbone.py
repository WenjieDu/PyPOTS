"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import torch.nn as nn


class BackboneTRMF(nn.Module):
    """The backbone of Temporal Regularized Matrix Factorization (TRMF).

    Parameters
    ----------
    lags : array-like, shape (n_lags,)
        Set of lag indices to use in model.

    K : int
        Length of latent embedding dimension

    lambda_f : float
        Regularization parameter used for matrix F.

    lambda_x : float
        Regularization parameter used for matrix X.

    lambda_w : float
        Regularization parameter used for matrix W.

    alpha : float
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.

    eta : float
        Regularization parameter used for X when undercovering autoregressive dependencies.

    max_iter : int
        Number of iterations of updating matrices F, X and W.

    F_step : float
        Step of gradient descent when updating matrix F.

    X_step : float
        Step of gradient descent when updating matrix X.

    W_step : float
        Step of gradient descent when updating matrix W.

    Attributes
    ----------

    F : ndarray, shape (n_timeseries, K)
        Latent embedding of timeseries.

    X : ndarray, shape (K, n_timepoints)
        Latent embedding of timepoints.

    W : ndarray, shape (K, n_lags)
        Matrix of autoregressive coefficients.

    """

    def __init__(
        self,
        lags,
        K,
        lambda_f,
        lambda_x,
        lambda_w,
        alpha,
        eta,
        max_iter,
        F_step=0.0001,
        X_step=0.0001,
        W_step=0.0001,
    ):
        super().__init__()

        self.lags = lags
        self.L = len(lags)
        self.K = K
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.lambda_w = lambda_w
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.F_step = F_step
        self.X_step = X_step
        self.W_step = W_step

        self.W = None
        self.F = None
        self.X = None

        self.Y = None
        self.mask = None
        self.N = None
        self.T = None

        # Flag to check if the model is trained.
        # It is useful when we want to continue training the model without initialization after it was trained.
        self.trained = False

    def forward(self, X, missing_mask):
        # Transpose the input data to have the same dimensionality as in the original TRMF implementation
        X = X.T
        missing_mask = missing_mask.T

        self.Y = X.copy()
        self.mask = missing_mask.copy()
        (
            self.N,
            self.T,
        ) = self.Y.shape

        # if not trained, initialize the matrices
        if not self.trained:
            self.W = np.random.randn(self.K, self.L) / self.L
            self.F = np.random.randn(self.N, self.K)
            self.X = np.random.randn(self.K, self.T)
            self.trained = True

        for _ in range(self.max_iter):
            self._update_F(step=self.F_step)
            self._update_X(step=self.X_step)
            self._update_W(step=self.W_step)

    def impute_missingness(self):
        """Impute each missing element in timeseries.

        Model uses matrix X and F to get all missing elements.

        Parameters
        ----------

        Returns
        -------
        data : ndarray, shape (n_timeseries, T)
            Imputed data.
        """
        data = self.Y
        data[self.mask == 0] = np.dot(self.F, self.X)[self.mask == 0]
        return data.T

    def _update_F(self, step, n_iter=1):
        """Gradient descent of matrix F.

        n_iter steps of gradient descent of matrix F.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.F -= step * self._grad_F()

    def _grad_F(self):
        """Gradient of matrix F.

        Evaluating gradient of matrix F.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """
        return -2 * np.dot((self.Y - np.dot(self.F, self.X)) * self.mask, self.X.T) + 2 * self.lambda_f * self.F

    def _update_X(self, step, n_iter=1):
        """Gradient descent of matrix X.

        n_iter steps of gradient descent of matrix X.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.X -= step * self._grad_X()

    def _grad_X(self):
        """Gradient of matrix X.

        Evaluating gradient of matrix X.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        for i in range(self.L):
            lag = self.lags[i]
            W_i = self.W[:, i].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_i = self.X * W_i
            z_1 = self.X - np.roll(X_i, lag, axis=1)
            z_1[:, : max(self.lags)] = 0.0
            z_2 = -(np.roll(self.X, -lag, axis=1) - X_i) * W_i
            z_2[:, -lag:] = 0.0

        grad_T_x = z_1 + z_2
        return (
            -2 * np.dot(self.F.T, self.mask * (self.Y - np.dot(self.F, self.X)))
            + self.lambda_x * grad_T_x
            + self.eta * self.X
        )

    def _update_W(self, step, n_iter=1):
        """Gradient descent of matrix W.

        n_iter steps of gradient descent of matrix W.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.W -= step * self._grad_W()

    def _grad_W(self):
        """Gradient of matrix W.

        Evaluating gradient of matrix W.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        grad = np.zeros((self.K, self.L))
        for i in range(self.L):
            lag = self.lags[i]
            W_i = self.W[:, i].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_i = self.X * W_i
            z_1 = self.X - np.roll(X_i, lag, axis=1)
            z_1[:, : max(self.lags)] = 0.0
            z_2 = -(z_1 * np.roll(self.X, lag, axis=1)).sum(axis=1)
            grad[:, i] = z_2
        return (
            grad
            + self.W * 2 * self.lambda_w / self.lambda_x
            - self.alpha * 2 * (1 - self.W.sum(axis=1)).repeat(self.L).reshape(self.W.shape)
        )
