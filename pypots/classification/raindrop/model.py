"""
The implementation of Raindrop for the partially-observed time-series classification task.

Refer to the paper "Zhang, X., Zeman, M., Tsiligkaridis, T., & Zitnik, M. (2022).
Graph-Guided Network for Irregularly Sampled Multivariate Time Series. ICLR 2022."

"""


# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .modules import _Raindrop
from ...classification.base import BaseNNClassifier
from ...classification.grud.data import DatasetForGRUD
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger


class Raindrop(BaseNNClassifier):
    """The PyTorch implementation of the Raindrop model :cite:`zhang2022Raindrop`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data samples.

    n_classes :
        The number of classes in the classification task.

    n_layers :
        The number of layers in the Transformer encoder in the Raindrop model.

    d_model :
        The dimension of the Transformer encoder backbone.
        It is the input dimension of the multi-head self-attention layers.

    d_inner :
        The dimension of the layer in the Feed-Forward Networks (FFN).

    n_heads :
        The number of heads in the multi-head self-attention mechanism.

    dropout :
        The dropout rate for all fully-connected layers in the model.

    d_static :
        The dimension of the static features.

    aggregation :
        The aggregation method for the Transformer encoder output.

    sensor_wise_mask :
        Whether to apply the sensor-wise masking.

    static :
        Whether to use the static features.

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

    References
    ----------
    .. [1] `Zhang, Xiang, Marko Zeman, Theodoros Tsiligkaridis, and Marinka Zitnik.
        "Graph-guided network for irregularly sampled multivariate time series."
        International Conference on Learning Representations (ICLR). 2022.
        <https://openreview.net/forum?id=Kwm8I7dU-l5>`_
    """

    def __init__(
        self,
        n_steps,
        n_features,
        n_classes,
        n_layers,
        d_model,
        d_inner,
        n_heads,
        dropout,
        d_static=0,
        aggregation="mean",
        sensor_wise_mask=False,
        static=False,
        batch_size=32,
        epochs=100,
        patience: Optional[int] = None,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
    ):
        super().__init__(
            n_classes,
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
        )

        self.n_features = n_features
        self.n_steps = n_steps

        # set up the model
        self.model = _Raindrop(
            n_features,
            n_layers,
            d_model,
            d_inner,
            n_heads,
            n_classes,
            dropout,
            n_steps,
            d_static,
            aggregation,
            sensor_wise_mask,
            static=static,
            device=self.device,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            X_filledLOCF,
            missing_mask,
            deltas,
            empirical_mean,
            label,
        ) = self._send_data_to_given_device(data)

        bz, n_steps, n_features = X.shape
        lengths = torch.tensor([n_steps] * bz, dtype=torch.float)
        times = torch.tensor(range(n_steps), dtype=torch.float).repeat(bz, 1)

        inputs = {
            "X": X,
            "static": None,
            "timestamps": times,
            "lengths": lengths,
            "missing_mask": missing_mask,
            "label": label,
        }
        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        (
            indices,
            X,
            X_filledLOCF,
            missing_mask,
            deltas,
            empirical_mean,
        ) = self._send_data_to_given_device(data)
        bz, n_steps, n_features = X.shape
        lengths = torch.tensor([n_steps] * bz, dtype=torch.float)
        times = torch.tensor(range(n_steps), dtype=torch.float).repeat(bz, 1)

        inputs = {
            "X": X,
            "static": None,
            "timestamps": times,
            "lengths": lengths,
            "missing_mask": missing_mask,
        }

        return inputs

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type="h5py",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForGRUD(train_set, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            val_set = DatasetForGRUD(val_set, file_type=file_type)
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
        self._auto_save_model_if_necessary(confirm_saving=True)

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "h5py",
    ) -> dict:
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGRUD(test_set, return_labels=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        classification_collector = []
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                prediction = results["classification_pred"]
                classification_collector.append(prediction)

        classification = torch.cat(classification_collector).cpu().detach().numpy()

        result_dict = {
            "classification": classification,
        }
        return result_dict

    def classify(
        self,
        X: Union[dict, str],
        file_type: str = "h5py",
    ) -> np.ndarray:
        """Classify the input data with the trained model.

        Warnings
        --------
        The method classify is deprecated. Please use `predict()` instead.

        Parameters
        ----------
        X :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples],
            Classification results of the given samples.
        """
        logger.warning(
            "🚨DeprecationWarning: The method classify is deprecated. Please use `predict` instead."
        )
        result_dict = self.predict(X, file_type=file_type)
        return result_dict["classification"]
