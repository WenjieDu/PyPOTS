"""
The implementation of CRLI (Clustering Representation Learning on Incomplete time-series data) for
the partially-observed time-series clustering task.

Refer to the paper "Ma, Q., Chen, C., Li, S., & Cottrell, G. W. (2021).
Learning Representations for Incomplete Time Series Clustering. AAAI 2021."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from pypots.clustering.base import BaseNNClusterer
from pypots.clustering.crli.data import DatasetForCRLI
from pypots.clustering.crli.modules import Generator, Decoder, Discriminator
from pypots.optim.adam import Adam
from pypots.optim.base import Optimizer
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mse


class _CRLI(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_clusters: int,
        n_generator_layers: int,
        rnn_hidden_size: int,
        decoder_fcn_output_dims: Optional[list],
        lambda_kmeans: float,
        rnn_cell_type: str = "GRU",
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.generator = Generator(
            n_generator_layers, n_features, rnn_hidden_size, rnn_cell_type, device
        )
        self.discriminator = Discriminator(rnn_cell_type, n_features, device)
        self.decoder = Decoder(
            n_steps, rnn_hidden_size * 2, n_features, decoder_fcn_output_dims, device
        )  # fully connected network is included in Decoder
        self.kmeans = KMeans(
            n_clusters=n_clusters
        )  # TODO: implement KMean with torch for gpu acceleration

        self.n_clusters = n_clusters
        self.lambda_kmeans = lambda_kmeans
        self.device = device

    def cluster(self, inputs: dict, training_object: str = "generator") -> dict:
        # concat final states from generator and input it as the initial state of decoder
        imputation, imputed_X, generator_fb_hidden_states = self.generator(inputs)
        inputs["imputation"] = imputation
        inputs["imputed_X"] = imputed_X
        inputs["generator_fb_hidden_states"] = generator_fb_hidden_states
        if training_object == "discriminator":
            discrimination = self.discriminator(inputs)
            inputs["discrimination"] = discrimination
            return inputs  # if only train discriminator, then no need to run decoder

        reconstruction, fcn_latent = self.decoder(inputs)
        inputs["reconstruction"] = reconstruction
        inputs["fcn_latent"] = fcn_latent
        return inputs

    def forward(self, inputs: dict, training_object: str = "generator") -> dict:
        assert training_object in [
            "generator",
            "discriminator",
        ], 'training_object should be "generator" or "discriminator"'

        X = inputs["X"]
        missing_mask = inputs["missing_mask"]
        batch_size, n_steps, n_features = X.shape
        losses = {}
        inputs = self.cluster(inputs, training_object)
        if training_object == "discriminator":
            l_D = F.binary_cross_entropy_with_logits(
                inputs["discrimination"], missing_mask
            )
            losses["discrimination_loss"] = l_D
        else:
            inputs["discrimination"] = inputs["discrimination"].detach()
            l_G = F.binary_cross_entropy_with_logits(
                inputs["discrimination"], 1 - missing_mask, weight=1 - missing_mask
            )
            l_pre = cal_mse(inputs["imputation"], X, missing_mask)
            l_rec = cal_mse(inputs["reconstruction"], X, missing_mask)
            HTH = torch.matmul(inputs["fcn_latent"], inputs["fcn_latent"].permute(1, 0))
            term_F = torch.nn.init.orthogonal_(
                torch.randn(batch_size, self.n_clusters, device=self.device), gain=1
            )
            FTHTHF = torch.matmul(torch.matmul(term_F.permute(1, 0), HTH), term_F)
            l_kmeans = torch.trace(HTH) - torch.trace(FTHTHF)  # k-means loss
            loss_gene = l_G + l_pre + l_rec + l_kmeans * self.lambda_kmeans
            losses["generation_loss"] = loss_gene
        return losses


class CRLI(BaseNNClusterer):
    """The PyTorch implementation of the CRLI model :cite:`ma2021CRLI`.

    Parameters
    ----------
    n_steps : int,
        The number of time steps in the time-series data sample.

    n_features : int,
        The number of features in the time-series data sample.

    n_clusters : int,
        The number of clusters in the clustering task.

    n_generator_layers : int,
        The number of layers in the generator.

    rnn_hidden_size : int,
        The size of the RNN hidden state, also the number of hidden units in the RNN cell.

    rnn_cell_type : str, default = "GRU"
        The type of RNN cell to use. Can be either "GRU" or "LSTM".

    decoder_fcn_output_dims : list, default = None
        The output dimensions of each layer in the FCN (fully-connected network) of the decoder.

    lambda_kmeans : float,
        The weight of the k-means loss,
        i.e. the item :math:`\\lambda` ahead of :math:`\\mathcal{L}_{k-means}` in Eq.13 of the original paper.

    G_steps : int, default = 1,
        The number of steps to train the generator in each iteration.

    D_steps : int, default = 1,
        The number of steps to train the discriminator in each iteration.

    batch_size : int, default = 32,
        The batch size for training and evaluating the model.

    epochs : int, default = 100,
        The number of epochs for training the model.

    patience : int, default = None,
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    G_optimizer : ``pypots.optim.base.Optimizer``, default = ``pypots.optim.Adam()``,
        The optimizer for the generator training.
        If not given, will use a default Adam optimizer.

    D_optimizer : ``pypots.optim.base.Optimizer``, default = ``pypots.optim.Adam()``,
        The optimizer for the discriminator training.
        If not given, will use a default Adam optimizer.

    num_workers : int, default = 0,
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device : str or `torch.device`, default = None,
        The device for the model to run on.
        If not given, will try to use CUDA devices first (will use the GPU with device number 0 only by default),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str, default = None,
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy : str or None, None or "best" or "better" , default = "best",
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    Attributes
    ----------
    model : object,
        The underlying CRLI model.

    optimizer : object,
        The optimizer for model training.

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
        G_optimizer: Optional[Optimizer] = Adam(),
        D_optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
    ):

        super().__init__(
            n_clusters,
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
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
            self.device,
        )
        self.model = self.model.to(self.device)
        self.print_model_size()

        # set up the optimizer
        self.G_optimizer = G_optimizer
        self.G_optimizer.init_optimizer(
            [
                {"params": self.model.generator.parameters()},
                {"params": self.model.decoder.parameters()},
            ]
        )
        self.D_optimizer = D_optimizer
        self.D_optimizer.init_optimizer(self.model.discriminator.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        """Assemble the given data into a dictionary for training input.

        Parameters
        ----------
        data : list,
            A list containing data fetched from Dataset by Dataloader.

        Returns
        -------
        inputs : dict,
            A python dictionary contains the input data for model training.
        """

        # fetch data
        indices, X, missing_mask = map(lambda x: x.to(self.device), data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        """Assemble the given data into a dictionary for validating input.

        Notes
        -----
        The validating data assembling processing is the same as training data assembling.


        Parameters
        ----------
        data : list,
            A list containing data fetched from Dataset by Dataloader.

        Returns
        -------
        inputs : dict,
            A python dictionary contains the input data for model validating.
        """
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        """Assemble the given data into a dictionary for testing input.

        Notes
        -----
        The testing data assembling processing is the same as training data assembling.

        Parameters
        ----------
        data : list,
            A list containing data fetched from Dataset by Dataloader.

        Returns
        -------
        inputs : dict,
            A python dictionary contains the input data for model testing.
        """
        return self._assemble_input_for_validating(data)

    def _train_model(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None

        try:
            training_step = 0
            epoch_train_loss_G_collector = []
            epoch_train_loss_D_collector = []
            for epoch in range(self.epochs):
                self.model.train()
                for idx, data in enumerate(training_loader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)

                    step_train_loss_G_collector = []
                    step_train_loss_D_collector = []
                    for _ in range(self.D_steps):
                        self.D_optimizer.zero_grad()
                        results = self.model.forward(
                            inputs, training_object="discriminator"
                        )
                        results["discrimination_loss"].backward(retain_graph=True)
                        self.D_optimizer.step()
                        step_train_loss_D_collector.append(
                            results["discrimination_loss"].item()
                        )

                    for _ in range(self.G_steps):
                        self.G_optimizer.zero_grad()
                        results = self.model.forward(
                            inputs, training_object="generator"
                        )
                        results["generation_loss"].backward()
                        self.G_optimizer.step()
                        step_train_loss_G_collector.append(
                            results["generation_loss"].item()
                        )

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
                        self._save_log_into_tb_file(
                            training_step, "training", loss_results
                        )
                mean_epoch_train_D_loss = np.mean(epoch_train_loss_D_collector)
                mean_epoch_train_G_loss = np.mean(epoch_train_loss_G_collector)
                logger.info(
                    f"epoch {epoch}: "
                    f"training loss_generator {mean_epoch_train_G_loss:.4f}, "
                    f"train loss_discriminator {mean_epoch_train_D_loss:.4f}"
                )
                mean_loss = mean_epoch_train_G_loss

                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                    # save the model if necessary
                    self._auto_save_model_if_necessary(
                        training_finished=False,
                        saving_name=f"{self.__class__.__name__}_epoch{epoch}_loss{mean_loss}",
                    )
                else:
                    self.patience -= 1
                    if self.patience == 0:
                        logger.info(
                            "Exceeded the training patience. Terminating the training procedure..."
                        )
                        break
        except Exception as e:
            logger.info(f"Exception: {e}")
            if self.best_model_dict is None:
                raise RuntimeError(
                    "Training got interrupted. Model was not get trained. Please try fit() again."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. "
                    "Model will load the best parameters so far for testing. "
                    "If you don't want it, please try fit() again."
                )

        if np.equal(self.best_loss, float("inf")):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info("Finished training.")

    def fit(
        self,
        train_set: Union[dict, str],
        file_type: str = "h5py",
    ) -> None:
        """Train the cluster.

        Parameters
        ----------
        train_set : dict or str,
            The dataset for model training, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for training, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        file_type : str, default = "h5py"
            The type of the given file if train_set is a path string.

        """
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForCRLI(
            train_set, return_labels=False, file_type=file_type
        )
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        # Step 2: train the model and freeze it
        self._train_model(training_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(training_finished=True)

    def cluster(
        self,
        X: Union[dict, str],
        file_type: str = "h5py",
    ) -> np.ndarray:
        """Cluster the input with the trained model.

        Parameters
        ----------
        X : array-like or str,
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type : str, default = "h5py"
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples],
            Clustering results.
        """
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForCRLI(X, return_labels=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        latent_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                inputs = self.model.cluster(inputs)
                latent_collector.append(inputs["fcn_latent"])

        latent_collector = torch.cat(latent_collector).cpu().detach().numpy()
        clustering = self.model.kmeans.fit_predict(latent_collector)

        return clustering
