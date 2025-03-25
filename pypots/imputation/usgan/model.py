"""
The implementation of USGAN for the partially-observed time-series imputation task.

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
from copy import deepcopy
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _USGAN
from ..base import BaseNNImputer
from ..brits.data import DatasetForBRITS
from ...data.checking import key_in_data_set
from ...nn.functional import calc_mse
from ...nn.functional import gather_listed_dicts
from ...nn.modules.loss import Criterion
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger

try:
    import nni
except ImportError:
    pass


class USGAN(BaseNNImputer):
    """The PyTorch implementation of the USGAN model. Refer to :cite:`miao2021SSGAN`.

    Parameters
    ----------
    n_steps : int
        The number of time steps in the time-series data sample.

    n_features : int
        The number of features in the time-series data sample.

    rnn_hidden_size : int
        The hidden size of the RNN cell

    lambda_mse : float
        The weight of the reconstruction loss

    hint_rate : float
        The hint rate for the discriminator

    dropout : float
        The dropout rate for the last layer in Discriminator

    G_steps : int
        The number of steps to train the generator in each iteration.

    D_steps : int
        The number of steps to train the discriminator in each iteration.

    batch_size : int
        The batch size for training and evaluating the model.

    epochs : int
        The number of epochs for training the model.

    patience : int
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    G_optimizer : :class:`pypots.optim.Optimizer`
        The optimizer for the generator training.
        If not given, will use a default Adam optimizer.

    D_optimizer : :class:`pypots.optim.Optimizer`
        The optimizer for the discriminator training.
        If not given, will use a default Adam optimizer.

    num_workers : int
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device : Union[str, torch.device, list]
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy : str
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        lambda_mse: float = 1,
        hint_rate: float = 0.7,
        dropout: float = 0.0,
        G_steps: int = 1,
        D_steps: int = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        G_optimizer: Union[Optimizer, type] = Adam,
        D_optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            training_loss=Criterion,
            validation_metric=Criterion,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )
        assert G_steps > 0 and D_steps > 0, "G_steps and D_steps should both >0"

        self.n_steps = n_steps
        self.n_features = n_features
        self.G_steps = G_steps
        self.D_steps = D_steps

        # set up the model
        self.model = _USGAN(
            n_steps,
            n_features,
            rnn_hidden_size,
            lambda_mse,
            hint_rate,
            dropout,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        if isinstance(G_optimizer, Optimizer):
            self.G_optimizer = G_optimizer
        else:
            self.G_optimizer = G_optimizer()  # instantiate the optimizer if it is a class
            assert isinstance(self.G_optimizer, Optimizer)
        if isinstance(D_optimizer, Optimizer):
            self.D_optimizer = D_optimizer
        else:
            self.D_optimizer = D_optimizer()  # instantiate the optimizer if it is a class
            assert isinstance(self.D_optimizer, Optimizer)

        if isinstance(self.device, list):
            self.G_optimizer.init_optimizer(self.model.module.backbone.generator.parameters())
            self.D_optimizer.init_optimizer(self.model.module.backbone.discriminator.parameters())
        else:
            self.G_optimizer.init_optimizer(self.model.backbone.generator.parameters())
            self.D_optimizer.init_optimizer(self.model.backbone.discriminator.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
            deltas,
            back_X,
            back_missing_mask,
            back_deltas,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
            },
            "backward": {
                "X": back_X,
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
            },
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
            deltas,
            back_X,
            back_missing_mask,
            back_deltas,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
            },
            "backward": {
                "X": back_X,
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
            },
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _train_model(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
    ) -> None:
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_model_dict = None

        if self.validation_metric.lower_better:
            self.best_loss = float("inf")
        else:
            self.best_loss = float("-inf")

        try:
            training_step = 0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                step_train_loss_G_collector = []
                step_train_loss_D_collector = []
                for idx, data in enumerate(train_dataloader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)

                    if idx % self.G_steps == 0:
                        self.G_optimizer.zero_grad()
                        results = self.model(inputs, training_object="generator")
                        loss = results["loss"].sum()
                        loss.backward()  # generation loss
                        self.G_optimizer.step()
                        step_train_loss_G_collector.append(loss.item())

                    if idx % self.D_steps == 0:
                        self.D_optimizer.zero_grad()
                        results = self.model(inputs, training_object="discriminator")
                        loss = results["loss"].sum()
                        loss.backward(retain_graph=True)  # discrimination loss
                        self.D_optimizer.step()
                        step_train_loss_D_collector.append(loss.item())

                    mean_step_train_D_loss = np.mean(step_train_loss_D_collector)
                    mean_step_train_G_loss = np.mean(step_train_loss_G_collector)

                    # save training loss logs into the tensorboard file for every step if in need
                    # Note: the `training_step` is not the actual number of steps that Discriminator and Generator get
                    # trained, the actual number should be D_steps*training_step and G_steps*training_step accordingly
                    if self.summary_writer is not None:
                        loss_results = {
                            "generation_loss": mean_step_train_G_loss,
                            "discrimination_loss": mean_step_train_D_loss,
                        }
                        self._save_log_into_tb_file(training_step, "training", loss_results)
                mean_epoch_train_D_loss = np.mean(step_train_loss_D_collector)
                mean_epoch_train_G_loss = np.mean(step_train_loss_G_collector)

                if val_dataloader is not None:
                    self.model.eval()
                    imputation_loss_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_dataloader):
                            inputs = self._assemble_input_for_validating(data)
                            results = self.model(inputs)
                            imputation_mse = (
                                calc_mse(
                                    results["imputation"],
                                    inputs["X_ori"],
                                    inputs["indicating_mask"],
                                )
                                .sum()
                                .detach()
                                .item()
                            )
                            imputation_loss_collector.append(imputation_mse)

                    mean_val_loss = np.mean(imputation_loss_collector)
                    # save validation loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "validating_loss": mean_val_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)
                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"generator training loss: {mean_epoch_train_G_loss:.4f}, "
                        f"discriminator training loss: {mean_epoch_train_D_loss:.4f}, "
                        f"validation loss: {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"generator training loss: {mean_epoch_train_G_loss:.4f}, "
                        f"discriminator training loss: {mean_epoch_train_D_loss:.4f}"
                    )
                    mean_loss = mean_epoch_train_G_loss

                if np.isnan(mean_loss):
                    logger.warning(f"‼️ Attention: got NaN loss in Epoch {epoch}. This may lead to unexpected errors.")

                if (self.validation_metric.lower_better and mean_loss < self.best_loss) or (
                    not self.validation_metric.lower_better and mean_loss > self.best_loss
                ):
                    self.best_epoch = epoch
                    self.best_loss = mean_loss
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    self.patience = self.original_patience
                else:
                    self.patience -= 1

                # save the model if necessary
                self._auto_save_model_if_necessary(
                    confirm_saving=self.best_epoch == epoch and self.model_saving_strategy == "better",
                    saving_name=f"{self.__class__.__name__}_epoch{epoch}_{self.validation_metric_name}{mean_loss:.4f}",
                )

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

        if np.isnan(self.best_loss) or self.best_loss.__eq__(float("inf")):
            raise ValueError("Something is wrong. best_loss is Nan/Inf after training.")

        logger.info(f"Finished training. The best model is from epoch#{self.best_epoch}.")

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        train_dataset = DatasetForBRITS(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_dataloader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_dataset = DatasetForBRITS(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        self.model.eval()  # set the model to evaluation mode
        test_dataset = DatasetForBRITS(test_set, return_X_ori=False, return_y=False, file_type=file_type)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Step 2: process the data with the model
        dict_result_collector = []
        for idx, data in enumerate(test_dataloader):
            inputs = self._assemble_input_for_testing(data)
            results = self.model(inputs)
            dict_result_collector.append(results)

        # Step 3: output collection and return
        result_dict = gather_listed_dicts(dict_result_collector)

        return result_dict
