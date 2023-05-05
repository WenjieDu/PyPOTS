"""
PyTorch Raindrop model. Refer :cite:`zhang2022Raindrop` for more information.
Inspired by the original implementation from https://github.com/mims-harvard/Raindrop

Notes
-----
Due to the original implementation puts too many useless arguments and is not elegant, I simply the code.
If you need a version of the original implementation, please refer to my previous commit here
https://github.com/WenjieDu/PyPOTS/blob/c381ad1853b465ebb918134d8bf6f6cf2996c9d3/pypots/classification/raindrop.py
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from pypots.classification.base import BaseNNClassifier
from pypots.classification.grud.dataset import DatasetForGRUD
from pypots.classification.raindrop.modules import (
    PositionalEncoding,
    ObservationPropagation,
)
from pypots.utils.logging import logger

try:
    from torch_geometric.nn.inits import glorot
except ImportError as e:
    logger.error(
        f"{e}\n"
        "torch_geometric is missing, "
        "please install it with 'pip install torch_geometric' or 'conda install -c pyg pyg'"
    )


class _Raindrop(nn.Module):
    def __init__(
        self,
        n_layers,
        n_features,
        d_model,
        d_inner,
        n_heads,
        n_classes,
        dropout=0.3,
        max_len=215,
        d_static=9,
        aggregation="mean",
        sensor_wise_mask=False,
        static=False,
        device=None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_features = n_features
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.dropout = dropout
        self.max_len = max_len
        self.d_static = d_static
        self.aggregation = aggregation
        self.sensor_wise_mask = sensor_wise_mask
        self.static = static
        self.device = device

        # create modules
        self.global_structure = torch.ones(n_features, n_features, device=self.device)
        if self.static:
            self.emb = nn.Linear(d_static, n_features)
        assert d_model % n_features == 0, "d_model must be divisible by n_features"
        self.d_ob = int(d_model / n_features)
        self.encoder = nn.Linear(n_features * self.d_ob, n_features * self.d_ob)
        d_pe = 16
        self.pos_encoder = PositionalEncoding(d_pe, max_len)
        if self.sensor_wise_mask:
            dim_check = n_features * (self.d_ob + d_pe)
            assert dim_check % n_heads == 0, "dim_check must be divisible by n_heads"
            encoder_layers = TransformerEncoderLayer(
                n_features * (self.d_ob + d_pe), n_heads, d_inner, dropout
            )
        else:
            dim_check = d_model + d_pe
            assert dim_check % n_heads == 0, "dim_check must be divisible by n_heads"
            encoder_layers = TransformerEncoderLayer(
                d_model + d_pe, n_heads, d_inner, dropout
            )
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.adj = torch.ones([self.n_features, self.n_features], device=self.device)

        self.R_u = Parameter(torch.Tensor(1, self.n_features * self.d_ob))

        self.ob_propagation = ObservationPropagation(
            in_channels=max_len * self.d_ob,
            out_channels=max_len * self.d_ob,
            heads=1,
            n_nodes=n_features,
            ob_dim=self.d_ob,
        )
        self.ob_propagation_layer2 = ObservationPropagation(
            in_channels=max_len * self.d_ob,
            out_channels=max_len * self.d_ob,
            heads=1,
            n_nodes=n_features,
            ob_dim=self.d_ob,
        )
        if static:
            d_final = d_model + d_pe + n_features
        else:
            d_final = d_model + d_pe

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init_range = 1e-10
        self.encoder.weight.data.uniform_(-init_range, init_range)
        if self.static:
            self.emb.weight.data.uniform_(-init_range, init_range)
        glorot(self.R_u)

    def classify(self, inputs: dict) -> torch.Tensor:
        """Forward processing of BRITS.

        Parameters
        ----------
        inputs : dict,
            The input data.

        Returns
        -------
        dict, A dictionary includes all results.
            X : array, shape of [n_steps, n_samples, n_features]
                Time series data. If samples have different lengths, then need to be padded first.
            static : array, shape of [n_samples, n_features]
                Static features.
            times : array, shape of [n_steps, n_samples]
                Timestamps of time series data.
            lengths : array, shape of [n_samples]
                Number of nonzero recordings.
            missing_mask : array, shape of [n_steps, n_samples, n_features]
        """
        src = inputs["X"]
        static = inputs["static"]
        times = inputs["timestamps"]
        lengths = inputs["lengths"]
        missing_mask = inputs["missing_mask"]

        max_len, batch_size = src.shape[0], src.shape[1]

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src * self.R_u)
        pe = self.pos_encoder(times).to(self.device)
        if static is not None:
            emb = self.emb(static)

        h = self.dropout(h)

        mask = torch.arange(max_len)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(self.device)

        x = h

        adj = self.global_structure
        adj[torch.eye(self.n_features, dtype=torch.bool)] = 1

        edge_index = torch.nonzero(adj).T
        edge_weights = adj[edge_index[0], edge_index[1]]

        batch_size = src.shape[1]
        n_step = src.shape[0]
        output = torch.zeros(
            [n_step, batch_size, self.n_features * self.d_ob], device=self.device
        )

        alpha_all = torch.zeros([edge_index.shape[1], batch_size], device=self.device)

        # iterate on each sample
        for unit in range(0, batch_size):
            step_data = x[:, unit, :]
            p_t = pe[:, unit, :]

            step_data = step_data.reshape([n_step, self.n_features, self.d_ob]).permute(
                1, 0, 2
            )
            step_data = step_data.reshape(self.n_features, n_step * self.d_ob)

            step_data, attention_weights = self.ob_propagation(
                step_data,
                p_t=p_t,
                edge_index=edge_index,
                edge_weights=edge_weights,
                use_beta=False,
                edge_attr=None,
                return_attention_weights=True,
            )

            edge_index_layer2 = attention_weights[0]
            edge_weights_layer2 = attention_weights[1].squeeze(-1)

            step_data, attention_weights = self.ob_propagation_layer2(
                step_data,
                p_t=p_t,
                edge_index=edge_index_layer2,
                edge_weights=edge_weights_layer2,
                use_beta=False,
                edge_attr=None,
                return_attention_weights=True,
            )

            step_data = step_data.view([self.n_features, n_step, self.d_ob])
            step_data = step_data.permute([1, 0, 2])  # [n_step, n_features, d_ob]
            step_data = step_data.reshape([-1, self.n_features * self.d_ob])

            output[:, unit, :] = step_data
            alpha_all[:, unit] = attention_weights[1].squeeze(-1)

        # distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
        # distance = torch.mean(distance)

        if self.sensor_wise_mask:
            extend_output = output.view(-1, batch_size, self.n_features, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.n_features, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.n_features * (self.d_ob + 16))
        else:
            output = torch.cat([output, pe], dim=2)

        r_out = self.transformer_encoder(output, src_key_padding_mask=mask)

        sensor_wise_mask = self.sensor_wise_mask

        lengths2 = lengths.unsqueeze(1).to(self.device)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        if sensor_wise_mask:
            output = torch.zeros(
                [batch_size, self.n_features, self.d_ob + 16], device=self.device
            )
            extended_missing_mask = missing_mask.view(-1, batch_size, self.n_features)
            for se in range(self.n_features):
                r_out = r_out.view(-1, batch_size, self.n_features, (self.d_ob + 16))
                out = r_out[:, :, se, :]
                l_ = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(
                    1
                )  # length
                out_sensor = torch.sum(
                    out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0
                ) / (l_ + 1)
                output[:, se, :] = out_sensor
            output = output.view([-1, self.n_features * (self.d_ob + 16)])
        elif self.aggregation == "mean":
            output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        else:
            raise RuntimeError

        if static is not None:
            output = torch.cat([output, emb], dim=1)

        logits = self.mlp_static(output)
        prediction = torch.softmax(logits, dim=1)

        return prediction

    def forward(self, inputs):
        prediction = self.classify(inputs)
        classification_loss = F.nll_loss(torch.log(prediction), inputs["label"])

        results = {
            "prediction": prediction,
            "loss": classification_loss
            # 'distance': distance,
        }

        return results


class Raindrop(BaseNNClassifier):
    """

    Parameters
    ----------

    learning_rate : float (0,1),
        The learning rate parameter for the optimizer.
    weight_decay : float in (0,1),
        The weight decay parameter for the optimizer.
    epochs : int,
        The number of training epochs.
    patience : int,
        The number of epochs with loss non-decreasing before early stopping the training.
    batch_size : int,
        The batch size of the training input.
    device :
        Run the model on which device.
    """

    def __init__(
        self,
        n_features,
        n_layers,
        d_model,
        d_inner,
        n_heads,
        n_classes,
        dropout,
        max_len,
        d_static,
        aggregation,
        sensor_wise_mask,
        static,
        batch_size=32,
        epochs=100,
        patience: int = None,
        learning_rate=1e-3,
        weight_decay=1e-5,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
    ):
        super().__init__(
            n_classes,
            batch_size,
            epochs,
            patience,
            learning_rate,
            weight_decay,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
        )

        self.n_features = n_features
        self.n_steps = max_len
        self.model = _Raindrop(
            n_layers,
            n_features,
            d_model,
            d_inner,
            n_heads,
            n_classes,
            dropout,
            max_len,
            d_static,
            aggregation,
            sensor_wise_mask,
            static=static,
            device=self.device,
        )
        self.model = self.model.to(self.device)
        self._print_model_size()

    def _assemble_input_for_training(self, data: dict) -> dict:
        """Assemble the input data into a dictionary.

        Parameters
        ----------
        data : list
            A list containing data fetched from Dataset by Dataload.

        Returns
        -------
        inputs : dict
            A dictionary with data assembled.
        """
        # fetch data
        indices, X, X_filledLOCF, missing_mask, deltas, empirical_mean, label = map(
            lambda x: x.to(self.device), data
        )

        bz, n_steps, n_features = X.shape
        lengths = torch.tensor([n_steps] * bz, dtype=torch.float)
        times = torch.tensor(range(n_steps), dtype=torch.float).repeat(bz, 1)

        X = X.permute(1, 0, 2)
        missing_mask = missing_mask.permute(1, 0, 2)
        times = times.permute(1, 0)

        inputs = {
            "X": X,
            "static": None,
            "timestamps": times,
            "lengths": lengths,
            "missing_mask": missing_mask,
            "label": label,
        }
        return inputs

    def _assemble_input_for_validating(self, data: dict) -> dict:
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

    def _assemble_input_for_testing(self, data: dict) -> dict:
        """Assemble the given data into a dictionary for testing input.

        Parameters
        ----------
        data : list,
            A list containing data fetched from Dataset by Dataloader.

        Returns
        -------
        inputs : dict,
            A python dictionary contains the input data for model testing.
        """
        indices, X, X_filledLOCF, missing_mask, deltas, empirical_mean = map(
            lambda x: x.to(self.device), data
        )
        bz, n_steps, n_features = X.shape
        lengths = torch.tensor([n_steps] * bz, dtype=torch.float)
        times = torch.tensor(range(n_steps), dtype=torch.float).repeat(bz, 1)

        X = X.permute(1, 0, 2)
        missing_mask = missing_mask.permute(1, 0, 2)
        times = times.permute(1, 0)

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
        """Fit the model on the given training data.

        Parameters
        ----------
        train_set : dict or str,
            The dataset for model training, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for training, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        val_set : dict or str,
            The dataset for model validating, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type : str, default = "h5py"
            The type of the given file if train_set and val_set are path strings.

        Returns
        -------
        self : object,
            Trained model.
        """
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
        self._auto_save_model_if_necessary(training_finished=True)

    def classify(self, X: Union[dict, str], file_type: str = "h5py") -> np.ndarray:
        """Classify the input data with the trained model.

        Parameters
        ----------
        X : array-like or str,
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type : str, default = "h5py",
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples],
            Classification results of the given samples.
        """
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGRUD(X, return_labels=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        prediction_collector = []
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                prediction = self.model.classify(inputs)
                prediction_collector.append(prediction)

        predictions = torch.cat(prediction_collector)
        return predictions.cpu().detach().numpy()
