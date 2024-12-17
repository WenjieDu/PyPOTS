"""
The implementation of TRMF for the partially-observed time-series imputation task,
which is mainly based on the implementation of TRMF in https://github.com/SemenovAlex/trmf/blob/master/trmf.py

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Union, Optional

import numpy as np
import torch
from pygrinder import fill_and_get_mask_numpy

from .core import _TRMF
from .data import DatasetForTRMF
from ..base import BaseImputer
from ...data import inverse_sliding_window, sliding_window
from ...data.dataset import BaseDataset
from ...utils.logging import logger


class TRMF(BaseImputer):
    """Temporal Regularized Matrix Factorization (TRMF) imputation method.

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

    saving_path :
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy :
        The strategy to save model checkpoints. It has to be one of [None, "best", "better", "all"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.
        The "all" strategy will save every model after each epoch training.

    verbose :
        Whether to print out the training logs during the training process.

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
        max_iter=1000,
        F_step=0.0001,
        X_step=0.0001,
        W_step=0.0001,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.model = _TRMF(
            lags,
            K,
            lambda_f,
            lambda_x,
            lambda_w,
            alpha,
            eta,
            max_iter,
            F_step,
            X_step,
            W_step,
        )

        logger.warning(
            "‼️Note that, as a traditional matrix factorization function, TRMF does not support validation set. "
            "Also, it only accepts 2-dim (time dim, feature dim) time series data, hence PyPOTS auto runs "
            "inverse_sliding_window func for your input in the unified format with 3-dim (sample dim, time dim, "
            "feature dim) and it assumes your samples window_len == sliding_len. If you generate samples "
            "using sliding_window func with window_len != sliding_len, it may produce non-ideal results."
        )

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForTRMF(train_set)
        if val_set is not None:
            raise RuntimeError("TRMF does not support validation set.")

        # Step 2: train the model and freeze it
        X = training_set.fetch_entire_dataset()["X"]
        X = inverse_sliding_window(X, training_set.n_steps)
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        X, missing_mask = fill_and_get_mask_numpy(X)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }

        self.model.forward(inputs)

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        diagonal_attention_mask: bool = True,
        return_latent_vars: bool = False,
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set :
            The dataset for model validating, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if test_set is a path string.

        diagonal_attention_mask :
            Whether to apply a diagonal attention mask to the self-attention mechanism in the testing stage.

        return_latent_vars :
            Whether to return the latent variables in SAITS, e.g. attention weights of two DMSA blocks and
            the weight matrix from the combination block, etc.

        Returns
        -------
        file_type :
            The dictionary containing the clustering results and latent variables if necessary.

        """
        # Step 1: wrap the input data with classes Dataset and DataLoader
        # self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(
            test_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )

        X = test_set.fetch_entire_dataset()["X"]
        X = inverse_sliding_window(X, test_set.n_steps)
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        X, missing_mask = fill_and_get_mask_numpy(X)
        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }

        # Step 3: output collection and return
        results = self.model.forward(inputs)
        imputation = sliding_window(results["imputed_data"], test_set.n_steps)
        result_dict = {
            "imputation": imputation,
        }

        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (n_steps), n_features],
            Imputed data.
        """

        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["imputation"]
