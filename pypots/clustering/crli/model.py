"""
The implementation of CRLI (Clustering Representation Learning on Incomplete time-series data) for
the partially-observed time-series clustering task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
from copy import deepcopy
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _CRLI
from .data import DatasetForCRLI
from ..base import BaseNNClusterer
from ...nn.modules.loss import Criterion
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger

try:
    import nni
except ImportError:
    pass


class CRLI(BaseNNClusterer):
    """The PyTorch implementation of the CRLI model :cite:`ma2021CRLI`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_clusters :
        The number of clusters in the clustering task.

    n_generator_layers :
        The number of layers in the generator.

    rnn_hidden_size :
        The size of the RNN hidden state, also the number of hidden units in the RNN cell.

    rnn_cell_type :
        The type of RNN cell to use. Can be either "GRU" or "LSTM".

    decoder_fcn_output_dims :
        The output dimensions of each layer in the FCN (fully-connected network) of the decoder.

    lambda_kmeans :
        The weight of the k-means loss,
        i.e. the item :math:`\\lambda` ahead of :math:`\\mathcal{L}_{k-means}` in Eq.13 of the original paper.

    G_steps :
        The number of steps to train the generator in each iteration.

    D_steps :
        The number of steps to train the discriminator in each iteration.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    G_optimizer :
        The optimizer for the generator training.
        If not given, will use a default Adam optimizer.

    D_optimizer :
        The optimizer for the discriminator training.
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
        n_clusters: int,
        n_generator_layers: int,
        rnn_hidden_size: int,
        rnn_cell_type: str = "GRU",
        lambda_kmeans: float = 1,
        decoder_fcn_output_dims: Optional[list] = None,
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
            n_clusters=n_clusters,
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
        self.model = _CRLI(
            n_steps,
            n_features,
            n_clusters,
            n_generator_layers,
            rnn_hidden_size,
            decoder_fcn_output_dims,
            lambda_kmeans,
            rnn_cell_type,
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
            self.G_optimizer.init_optimizer(
                [
                    {"params": self.model.module.backbone.generator.parameters()},
                    {"params": self.model.module.backbone.decoder.parameters()},
                ]
            )
            self.D_optimizer.init_optimizer(self.model.module.backbone.discriminator.parameters())
        else:
            self.G_optimizer.init_optimizer(
                [
                    {"params": self.model.backbone.generator.parameters()},
                    {"params": self.model.backbone.decoder.parameters()},
                ]
            )
            self.D_optimizer.init_optimizer(self.model.backbone.discriminator.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        indices, X, missing_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_validating(data)

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
            epoch_train_loss_G_collector = []
            epoch_train_loss_D_collector = []
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                for idx, data in enumerate(train_dataloader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)

                    step_train_loss_G_collector = []
                    step_train_loss_D_collector = []
                    for _ in range(self.D_steps):
                        self.D_optimizer.zero_grad()
                        results = self.model(inputs, training_object="discriminator")
                        discrimination_loss = results["discrimination_loss"].sum()
                        discrimination_loss.backward(retain_graph=True)
                        self.D_optimizer.step()
                        step_train_loss_D_collector.append(discrimination_loss.sum().item())

                    for _ in range(self.G_steps):
                        self.G_optimizer.zero_grad()
                        results = self.model(inputs, training_object="generator")
                        generation_loss = results["generation_loss"].sum()
                        generation_loss.backward()
                        self.G_optimizer.step()
                        step_train_loss_G_collector.append(generation_loss.sum().item())

                    mean_step_train_D_loss = np.mean(step_train_loss_D_collector)
                    mean_step_train_G_loss = np.mean(step_train_loss_G_collector)

                    epoch_train_loss_D_collector.append(mean_step_train_D_loss)
                    epoch_train_loss_G_collector.append(mean_step_train_G_loss)

                    # save training loss logs into the tensorboard file for every step if in need
                    # Note: the `training_step` is not the actual number of steps that Discriminator and Generator get
                    # trained, the actual number should be D_steps*training_step and G_steps*training_step accordingly
                    if self.summary_writer is not None:
                        loss_results = {
                            "generation_loss": mean_step_train_G_loss,
                            "discrimination_loss": mean_step_train_D_loss,
                        }
                        self._save_log_into_tb_file(training_step, "training", loss_results)

                mean_epoch_train_D_loss = np.mean(epoch_train_loss_D_collector)
                mean_epoch_train_G_loss = np.mean(epoch_train_loss_G_collector)

                if val_dataloader is not None:
                    self.model.eval()
                    epoch_val_loss_G_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_dataloader):
                            inputs = self._assemble_input_for_validating(data)
                            results = self.model(inputs)
                            generation_loss = results["generation_loss"]
                            epoch_val_loss_G_collector.append(generation_loss.sum().item())
                    mean_val_G_loss = np.mean(epoch_val_loss_G_collector)
                    # save validation loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "generation_loss": mean_val_G_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)
                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"generator training loss: {mean_epoch_train_G_loss:.4f}, "
                        f"discriminator training loss: {mean_epoch_train_D_loss:.4f}, "
                        f"generator validation loss: {mean_val_G_loss:.4f}"
                    )
                    mean_loss = mean_val_G_loss
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

                if os.getenv("ENABLE_HPO", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                # save the model if necessary
                self._auto_save_model_if_necessary(
                    confirm_saving=self.best_epoch == epoch and self.model_saving_strategy == "better",
                    saving_name=f"{self.__class__.__name__}_epoch{epoch}_{self.validation_metric_name}{mean_loss:.4f}",
                )

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
        train_dataset = DatasetForCRLI(train_set, return_y=False, file_type=file_type)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_dataloader = None

        if val_set is not None:
            val_dataset = DatasetForCRLI(val_set, return_y=False, file_type=file_type)
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
        return_latent_vars: bool = False,
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

        return_latent_vars : bool
            Whether to return the latent variables in VaDER, e.g. mu and phi, etc.

        Returns
        -------
        result_dict :
            The dictionary containing the clustering results as key 'clustering' and latent variables if necessary.
        """
        self.model.eval()  # set the model to evaluation mode
        test_dataset = DatasetForCRLI(
            test_set,
            return_y=False,
            file_type=file_type,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        clustering_latent_collector = []
        imputation_collector = []

        for idx, data in enumerate(test_dataloader):
            inputs = self._assemble_input_for_testing(data)
            inputs = self.model(inputs)
            clustering_latent_collector.append(inputs["fcn_latent"])
            if return_latent_vars:
                imputation_collector.append(inputs["imputation_latent"])

        clustering_latent = torch.cat(clustering_latent_collector).cpu().detach().numpy()
        if isinstance(self.device, list):
            clustering = self.model.module.kmeans.fit_predict(clustering_latent)
        else:
            clustering = self.model.kmeans.fit_predict(clustering_latent)

        result_dict = {
            "clustering": clustering,
        }

        if return_latent_vars:
            imputation = torch.cat(imputation_collector).cpu().detach().numpy()
            latent_var_collector = {
                "clustering_latent": clustering_latent,
                "imputation_latent": imputation,
            }
            result_dict["latent_vars"] = latent_var_collector

        return result_dict
