"""
The package of the partially-observed time-series imputation model ImputeFormer.

"""

# Created by Tong Nie <nietong@tongji.edu.cn> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _ImputeFormer
from .data import DatasetForImputeFormer
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...data.dataset import BaseDataset
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger


class ImputeFormer(BaseNNImputer):
    """The PyTorch implementation of the ImputeFormer model.
    ImputeFormer is originally proposed by Nie et al. in KDD'24: cite:`nie2024imputeformer`.


    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_layers :
        The number of layers in the 1st and 2nd DMSA blocks in the SAITS model.

    d_input_embed :
        The dimension of the input embedding.
        It is the input dimension of the input embedding layer.

    d_learnable_embed :
        The dimension of the learnable node embedding.
        It is the dimension of the learnable node embedding (spatial positional embedding)
        used in spatial attention layers.

    d_proj :
        The dimension of the learnable projector.
        It is the dimension of the learnable projector
        used in temporal attention layers.

    d_ffn :
        The dimension of the layer in the Feed-Forward Networks (FFN).

    dropout :
        The dropout rate for all fully-connected layers in the model.

    n_temporal_heads :
        The number of attention heads in temporal attention layers.

    input_dim :
        The dimension of the input feature dimension, default is 1.

    output_dim :
        The dimension of the output feature dimension, default is 1.

    ORT_weight :
        The weight for the ORT loss.

    MIT_weight :
        The weight for the MIT loss.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    optimizer :
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers :
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device :
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

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
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_input_embed: int,
        d_learnable_embed: int,
        d_proj: int,
        d_ffn: int,
        n_temporal_heads: int,
        dropout: float = 0.0,
        input_dim: int = 1,
        output_dim: int = 1,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        train_loss_func: Optional[dict] = None,
        val_metric_func: Optional[dict] = None,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            train_loss_func=train_loss_func,
            val_metric_func=val_metric_func,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        # model hype-parameters
        self.n_layers = n_layers
        self.d_input_embed = d_input_embed
        self.d_learnable_embed = d_learnable_embed
        self.d_proj = d_proj
        self.d_ffn = d_ffn
        self.n_temporal_heads = n_temporal_heads
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # set up the model
        self.model = _ImputeFormer(
            self.n_steps,
            self.n_features,
            self.n_layers,
            self.d_input_embed,
            self.d_learnable_embed,
            self.d_proj,
            self.d_ffn,
            self.n_temporal_heads,
            self.dropout,
            self.input_dim,
            self.output_dim,
            self.ORT_weight,
            self.MIT_weight,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        indices, X, missing_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }

        return inputs

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForImputeFormer(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForImputeFormer(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(
            test_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs)
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)

        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }
        return result_dict

    def impute(
        self,
        X: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Warnings
        --------
        The method impute is deprecated. Please use `predict()` instead.

        Parameters
        ----------
        X :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (time steps), n_features],
            Imputed data.
        """
        logger.warning("🚨DeprecationWarning: The method impute is deprecated. Please use `predict` instead.")
        results_dict = self.predict(X, file_type=file_type)
        return results_dict["imputation"]
