"""
Torch implementation of CRLI (Clustering Representation Learning on Incomplete time-series data).

Please refer to :cite:``ma2021CRLI``.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from pypots.clustering.base import BaseNNClusterer
from pypots.data.dataset_for_grud import DatasetForGRUD
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mse

RNN_CELL = {
    "LSTM": nn.LSTMCell,
    "GRU": nn.GRUCell,
}


def reverse_tensor(tensor_: torch.Tensor) -> torch.Tensor:
    if tensor_.dim() <= 1:
        return tensor_
    indices = range(tensor_.size()[1])[::-1]
    indices = torch.tensor(
        indices, dtype=torch.long, device=tensor_.device, requires_grad=False
    )
    return tensor_.index_select(1, indices)


class MultiRNNCell(nn.Module):
    def __init__(
        self,
        cell_type: str,
        n_layer: int,
        d_input: int,
        d_hidden: int,
        device: Union[str, torch.device],
    ):
        super().__init__()
        self.cell_type = cell_type
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.device = device

        self.model = nn.ModuleList()
        if cell_type in ["LSTM", "GRU"]:
            for i in range(n_layer):
                if i == 0:
                    self.model.append(RNN_CELL[cell_type](d_input, d_hidden))
                else:
                    self.model.append(RNN_CELL[cell_type](d_hidden, d_hidden))

        self.output_layer = nn.Linear(d_hidden, d_input)

    def forward(self, inputs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        bz, n_steps, _ = X.shape
        hidden_state = torch.zeros((bz, self.d_hidden), device=self.device)
        hidden_state_collector = torch.empty(
            (bz, n_steps, self.d_hidden), device=self.device
        )
        output_collector = torch.empty((bz, n_steps, self.d_input), device=self.device)
        if self.cell_type == "LSTM":
            # TODO: cell states should have different shapes
            cell_states = torch.zeros((self.d_input, self.d_hidden), device=self.device)
            for step in range(n_steps):
                x = X[:, step, :]
                estimation = self.output_layer(hidden_state)
                output_collector[:, step] = estimation
                imputed_x = (
                    missing_mask[:, step] * x + (1 - missing_mask[:, step]) * estimation
                )
                for i in range(self.n_layer):
                    if i == 0:
                        hidden_state, cell_states = self.model[i](
                            imputed_x, (hidden_state, cell_states)
                        )
                    else:
                        hidden_state, cell_states = self.model[i](
                            hidden_state, (hidden_state, cell_states)
                        )
                hidden_state_collector[:, step, :] = hidden_state

        elif self.cell_type == "GRU":
            for step in range(n_steps):
                x = X[:, step, :]
                estimation = self.output_layer(hidden_state)
                output_collector[:, step] = estimation
                imputed_x = (
                    missing_mask[:, step] * x + (1 - missing_mask[:, step]) * estimation
                )
                for i in range(self.n_layer):
                    if i == 0:
                        hidden_state = self.model[i](imputed_x, hidden_state)
                    else:
                        hidden_state = self.model[i](hidden_state, hidden_state)

                hidden_state_collector[:, step, :] = hidden_state

        output_collector = output_collector[:, 1:]
        estimation = self.output_layer(hidden_state).unsqueeze(1)
        output_collector = torch.concat([output_collector, estimation], dim=1)
        return output_collector, hidden_state


class Generator(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_features: int,
        d_hidden: int,
        cell_type: str,
        device: Union[str, torch.device],
    ):
        super().__init__()
        self.f_rnn = MultiRNNCell(cell_type, n_layers, n_features, d_hidden, device)
        self.b_rnn = MultiRNNCell(cell_type, n_layers, n_features, d_hidden, device)

    def forward(self, inputs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_outputs, f_final_hidden_state = self.f_rnn(inputs)
        b_outputs, b_final_hidden_state = self.b_rnn(inputs)
        b_outputs = reverse_tensor(b_outputs)  # reverse the output of the backward rnn
        imputation = (f_outputs + b_outputs) / 2
        imputed_X = inputs["X"] * inputs["missing_mask"] + imputation * (
            1 - inputs["missing_mask"]
        )
        fb_final_hidden_states = torch.concat(
            [f_final_hidden_state, b_final_hidden_state], dim=-1
        )
        return imputation, imputed_X, fb_final_hidden_states


class Discriminator(nn.Module):
    def __init__(
        self,
        cell_type: str,
        d_input: int,
        device: Union[str, torch.device],
    ):
        super().__init__()
        self.cell_type = cell_type
        self.device = device
        # this setting is the same with the official implementation
        self.rnn_cell_module_list = nn.ModuleList(
            [
                RNN_CELL[cell_type](d_input, 32),
                RNN_CELL[cell_type](32, 16),
                RNN_CELL[cell_type](16, 8),
                RNN_CELL[cell_type](8, 16),
                RNN_CELL[cell_type](16, 32),
            ]
        )
        self.output_layer = nn.Linear(32, d_input)

    def forward(self, inputs: dict) -> torch.Tensor:
        imputed_X = inputs["imputed_X"]
        bz, n_steps, _ = imputed_X.shape
        hidden_states = [
            torch.zeros((bz, 32), device=self.device),
            torch.zeros((bz, 16), device=self.device),
            torch.zeros((bz, 8), device=self.device),
            torch.zeros((bz, 16), device=self.device),
            torch.zeros((bz, 32), device=self.device),
        ]
        hidden_state_collector = torch.empty((bz, n_steps, 32), device=self.device)
        if self.cell_type == "LSTM":
            cell_states = torch.zeros((self.d_input, self.d_hidden), device=self.device)
            for step in range(n_steps):
                x = imputed_X[:, step, :]
                for i, rnn_cell in enumerate(self.rnn_cell_module_list):
                    if i == 0:
                        hidden_state, cell_states = rnn_cell(
                            x, (hidden_states[i], cell_states)
                        )
                    else:
                        hidden_state, cell_states = rnn_cell(
                            hidden_states[i - 1], (hidden_states[i], cell_states)
                        )
                    hidden_states[i] = hidden_state
                hidden_state_collector[:, step, :] = hidden_state

        elif self.cell_type == "GRU":
            for step in range(n_steps):
                x = imputed_X[:, step, :]
                for i, rnn_cell in enumerate(self.rnn_cell_module_list):
                    if i == 0:
                        hidden_state = rnn_cell(x, hidden_states[i])
                    else:
                        hidden_state = rnn_cell(hidden_states[i - 1], hidden_states[i])
                    hidden_states[i] = hidden_state
                hidden_state_collector[:, step, :] = hidden_state

        output_collector = self.output_layer(hidden_state_collector)
        return output_collector


class Decoder(nn.Module):
    def __init__(
        self,
        n_steps: int,
        d_input: int,
        d_output: int,
        fcn_output_dims: list = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.n_steps = n_steps
        self.d_output = d_output
        self.device = device

        if fcn_output_dims is None:
            fcn_output_dims = [d_input]
        self.fcn_output_dims = fcn_output_dims

        self.fcn = nn.ModuleList()
        for output_dim in fcn_output_dims:
            self.fcn.append(nn.Linear(d_input, output_dim))
            d_input = output_dim

        self.rnn_cell = nn.GRUCell(fcn_output_dims[-1], fcn_output_dims[-1])
        self.output_layer = nn.Linear(fcn_output_dims[-1], d_output)

    def forward(self, inputs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        generator_fb_hidden_states = inputs["generator_fb_hidden_states"]
        bz, _ = generator_fb_hidden_states.shape
        fcn_latent = generator_fb_hidden_states
        for layer in self.fcn:
            fcn_latent = layer(fcn_latent)
        hidden_state = fcn_latent
        hidden_state_collector = torch.empty(
            (bz, self.n_steps, self.fcn_output_dims[-1]), device=self.device
        )
        for i in range(self.n_steps):
            hidden_state = self.rnn_cell(hidden_state, hidden_state)
            hidden_state_collector[:, i, :] = hidden_state
        reconstruction = self.output_layer(hidden_state_collector)
        return reconstruction, fcn_latent


class _CRLI(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_clusters: int,
        n_generator_layers: int,
        rnn_hidden_size: int,
        decoder_fcn_output_dims: list,
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
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_clusters: int,
        n_generator_layers: int,
        rnn_hidden_size: int,
        decoder_fcn_output_dims: list = None,
        lambda_kmeans: float = 1,
        rnn_cell_type: str = "GRU",
        G_steps: int = 1,
        D_steps: int = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        tb_file_saving_path: str = None,
    ):
        super().__init__(
            n_clusters,
            batch_size,
            epochs,
            patience,
            learning_rate,
            weight_decay,
            num_workers,
            device,
            tb_file_saving_path,
        )
        assert G_steps > 0 and D_steps > 0, "G_steps and D_steps should both >0"

        self.n_steps = n_steps
        self.n_features = n_features
        self.G_steps = G_steps
        self.D_steps = D_steps

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
        self._print_model_size()

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
        indices, X, _, missing_mask, _, _ = map(lambda x: x.to(self.device), data)

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
        self.G_optimizer = torch.optim.Adam(
            [
                {"params": self.model.generator.parameters()},
                {"params": self.model.decoder.parameters()},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.D_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

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
                        self.save_log_into_tb_file(
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
        training_set = DatasetForGRUD(train_set, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self._train_model(training_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

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
        test_set = DatasetForGRUD(X, return_labels=False, file_type=file_type)
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
