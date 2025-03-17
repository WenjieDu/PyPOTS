"""
The implementation of TS2Vec for the partially-observed time-series classification task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import warnings
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DatasetForTS2Vec
from ..base import BaseNNClassifier
from ...nn.functional.cuda import autocast
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger
from ...vec.ts2vec.core import _TS2Vec

try:
    import nni
except ImportError:
    pass

SUPPORTED_CLASSIFIERS = ["linear_regression", "svm", "knn"]


class TS2Vec(BaseNNClassifier):
    """The PyTorch implementation of the TS2Vec model :cite:`yue2022ts2vec`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_classes :
        The number of classes in the classification task.

    n_output_dims :
        The number of output dimensions for the vectorization of the time-series data sample.

    d_hidden :
        The number of hidden dimensions for the TS2VEC encoder.

    n_layers :
        The number of layers for the TS2VEC encoder.

    mask_mode :
        The mode for generating the mask for the TS2VEC encoder.
        It has to be one of ['binomial', 'continuous', 'all_true', 'all_false', 'mask_last'].

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
        n_classes: int,
        n_output_dims: int,
        d_hidden: int,
        n_layers: int,
        mask_mode: str = "binomial",
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        optimizer: Optimizer = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            n_classes=n_classes,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_output_dims = n_output_dims
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.mask_mode = mask_mode
        self.training_set_loader = None

        # set up the model
        self.model = _TS2Vec(
            self.n_steps,
            self.n_features,
            self.n_output_dims,
            self.d_hidden,
            self.n_layers,
            self.mask_mode,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

        self.training_loss_name = "default"
        self.validation_metric_name = "default loss"

    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
            y,
        ) = self._send_data_to_given_device(data)
        missing_mask = missing_mask.to(torch.bool)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": torch.masked_fill(X, ~missing_mask, torch.nan),
            "y": y,
        }
        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
        ) = self._send_data_to_given_device(data)
        missing_mask = missing_mask.to(torch.bool)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": torch.masked_fill(X, ~missing_mask, torch.nan),
        }
        return inputs

    def _train_model(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        """

        Parameters
        ----------
        training_loader
        val_loader

        Notes
        -----
        The training procedures of NN clustering models are very different from each other. For example, VaDER needs
        pretraining while CRLI doesn't, VaDER only needs one optimizer while CRLI needs two for its generator and
        discriminator separately. So far, I'd suggest to implement function _train_model() for each model individually.

        """
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None

        try:
            training_step = 0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                epoch_train_loss_collector = []
                for idx, data in enumerate(training_loader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)

                    # model forward propagation processing
                    with autocast(enabled=self.amp_enabled):
                        self.optimizer.zero_grad()
                        results = self.model.forward(inputs)
                        results["loss"].sum().backward()  # sum() before backward() in case of multi-gpu training
                        self.optimizer.step()
                    epoch_train_loss_collector.append(results["loss"].sum().item())

                    # save training loss logs into the tensorboard file for every step if in need
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(training_step, "training", results)

                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_loader is not None:
                    self.model.eval()
                    epoch_val_loss_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self._assemble_input_for_validating(data)

                            # model forward propagation processing
                            with autocast(enabled=self.amp_enabled):
                                results = self.model.forward(inputs)
                            epoch_val_loss_collector.append(results["loss"].sum().item())

                    mean_val_loss = np.mean(epoch_val_loss_collector)
                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"training loss ({self.training_loss_name}): {mean_train_loss:.4f}, "
                        f"validation {self.validation_metric_name}: {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(f"Epoch {epoch:03d} - training loss ({self.training_loss_name}): {mean_train_loss:.4f}")
                    mean_loss = mean_train_loss

                if np.isnan(mean_loss):
                    logger.warning(f"‼️ Attention: got NaN loss in Epoch {epoch}. This may lead to unexpected errors.")

                if mean_loss < self.best_loss:
                    self.best_epoch = epoch
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                else:
                    self.patience -= 1

                if os.getenv("ENABLE_HPO", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.patience == 0:
                    logger.info("Exceeded the training patience. Terminating the training procedure...")
                    break

        except KeyboardInterrupt:  # if keyboard interrupt, only warning
            logger.warning("‼️ Training got interrupted by the user. Exist now ...")
        except Exception as e:  # other kind of exception follows below processing
            logger.error(f"❌ Exception: {e}")
            if self.best_model_dict is None:  # if no best model, raise error
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. Please investigate the error printed above."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
                    "If you don't want it, please try fit() again."
                )

        if np.isnan(self.best_loss):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info(f"Finished training. The best model is from epoch#{self.best_epoch}.")

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForTS2Vec(train_set, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.training_set_loader = training_loader
        val_loader = None
        if val_set is not None:
            val_set = DatasetForTS2Vec(val_set, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        classifier_type: str = "svm",
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set :
            The test dataset for model to process, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is the time-series data for processing.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include 'X' key.

        file_type :
            The type of the given file if test_set is a path string.

        classifier_type :
            The type of classifier to use for the classification task.
            It has to be one of ['linear_regression', 'svm', 'knn'].

        Returns
        -------
        file_type :
            The dictionary containing the clustering results and latent variables if necessary.

        """
        assert (
            classifier_type in SUPPORTED_CLASSIFIERS
        ), f"classifier_type should be one of {SUPPORTED_CLASSIFIERS}, but got {classifier_type}"

        self.model.eval()  # set the model to evaluation mode

        train_repr_collector = []
        train_label_collector = []
        for idx, data in enumerate(self.training_set_loader):
            inputs = self._assemble_input_for_training(data)
            train_repr = self.model(inputs, encoding_window="full_series")["representation"]
            train_repr_collector.append(train_repr)
            train_label_collector.append(inputs["y"])

        train_repr_collector = torch.cat(train_repr_collector, dim=0).cpu().numpy()
        train_label_collector = torch.cat(train_label_collector, dim=0).cpu().numpy()

        test_set = DatasetForTS2Vec(
            test_set,
            return_y=False,
            file_type=file_type,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        test_repr_collector = []
        for idx, data in enumerate(test_loader):
            inputs = self._assemble_input_for_testing(data)
            test_repr = self.model(inputs, encoding_window="full_series")["representation"]
            test_repr_collector.append(test_repr)

        test_repr_collector = torch.cat(test_repr_collector, dim=0).cpu().numpy()

        if classifier_type == "linear_regression":
            fit_clf = self.model.encoder.fit_lr
        elif classifier_type == "svm":
            fit_clf = self.model.encoder.fit_svm
        elif classifier_type == "knn":
            fit_clf = self.model.encoder.fit_knn
        else:
            raise ValueError()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")  # just ignore warnings, most of them from sklearn

            clf = fit_clf(train_repr_collector, train_label_collector)
            if classifier_type == "svm":
                y_score = clf.decision_function(test_repr_collector)
            else:
                y_score = clf.predict_proba(test_repr_collector)
            y_pred = clf.predict(test_repr_collector)

        result_dict = {
            "classification": y_pred,
            "score": y_score,
        }
        return result_dict

    def classify(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        classifier_type: str = "svm",
    ) -> np.ndarray:
        """Classify the input data with the trained model.

        Parameters
        ----------
        test_set :
            The test dataset for model to process, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is the time-series data for processing.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include 'X' key.

        file_type :
            The type of the given file if test_set is a path string.

        classifier_type :
            The type of classifier to use for the classification task.
            It has to be one of ['linear_regression', 'svm', 'knn'].

        Returns
        -------
        file_type :
            The dictionary containing the clustering results and latent variables if necessary.

        """
        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["classification"]
