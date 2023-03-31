"""
PyTorch SAITS model for the time-series imputation task.
Some part of the code is from https://github.com/WenjieDu/SAITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pypots.data.base import BaseDataset
from pypots.data.dataset_for_mit import DatasetForMIT
from pypots.data.integration import mcar, masked_fill
from pypots.imputation.base import BaseNNImputer
from pypots.imputation.transformer import EncoderLayer, PositionalEncoding
from pypots.utils.metrics import cal_mae


class _SAITS(nn.Module):
    def __init__(
        self,
        n_layers,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        diagonal_attention_mask=True,
        ORT_weight=1,
        MIT_weight=1,
    ):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.layer_stack_for_first_block = nn.ModuleList(
            [
                EncoderLayer(
                    d_time,
                    actual_d_feature,
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout,
                    0,
                    diagonal_attention_mask,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_stack_for_second_block = nn.ModuleList(
            [
                EncoderLayer(
                    d_time,
                    actual_d_feature,
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout,
                    0,
                    diagonal_attention_mask,
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]
        # first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2)
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely, term e in the math equation
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        input_X_for_second = torch.cat([X_prime, masks], dim=2)
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = torch.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = (
            masks * X + (1 - masks) * X_tilde_3
        )  # replace non-missing part with original data
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]
        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)

        reconstruction_loss += cal_mae(X_tilde_1, X, masks)
        reconstruction_loss += cal_mae(X_tilde_2, X, masks)
        final_reconstruction_MAE = cal_mae(X_tilde_3, X, masks)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        # have to cal imputation loss in the val stage; no need to cal imputation loss here in the tests stage
        imputation_loss = cal_mae(
            X_tilde_3, inputs["X_intact"], inputs["indicating_mask"]
        )

        loss = self.ORT_weight * reconstruction_loss + self.MIT_weight * imputation_loss

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_loss,
            "imputation_loss": imputation_loss,
            "loss": loss,
        }


class SAITS(BaseNNImputer):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        diagonal_attention_mask=True,
        ORT_weight=1,
        MIT_weight=1,
        learning_rate=1e-3,
        epochs=100,
        patience=10,
        batch_size=32,
        weight_decay=1e-5,
        device=None,
    ):
        super().__init__(
            learning_rate, epochs, patience, batch_size, weight_decay, device
        )

        self.n_steps = n_steps
        self.n_features = n_features
        # model hype-parameters
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.model = _SAITS(
            self.n_layers,
            self.n_steps,
            self.n_features,
            self.d_model,
            self.d_inner,
            self.n_head,
            self.d_k,
            self.d_v,
            self.dropout,
            self.diagonal_attention_mask,
            self.ORT_weight,
            self.MIT_weight,
        )
        self.model = self.model.to(self.device)
        self._print_model_size()

    def fit(self, train_set, val_set=None, file_type="h5py"):
        """Train the imputer on the given data.

        Parameters
        ----------
        train_set : dict or str,
            The dataset for model training, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for training, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        val_set : dict or str,
            The dataset for model validating, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for validating, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        file_type : str, default = "h5py",
            The type of the given file if train_set and val_set are path strings.

        Returns
        -------
        self : object,
            The trained imputer.
        """
        training_set = DatasetForMIT(train_set, file_type)
        training_loader = DataLoader(
            training_set, batch_size=self.batch_size, shuffle=True
        )
        if val_set is None:
            self._train_model(training_loader)
        else:
            if isinstance(val_set, str):
                import h5py

                with h5py.File(val_set, "r") as hf:
                    val_X = hf["X"][:]
                val_set = {"X": val_X}

            val_X_intact, val_X, val_X_missing_mask, val_X_indicating_mask = mcar(
                val_set["X"], 0.2
            )
            val_X = masked_fill(val_X, 1 - val_X_missing_mask, np.nan)
            val_set["X"] = val_X
            val_set = BaseDataset(val_set)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            self._train_model(
                training_loader, val_loader, val_X_intact, val_X_indicating_mask
            )

        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

    def assemble_input_for_training(self, data):
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

        indices, X_intact, X, missing_mask, indicating_mask = data

        inputs = {
            "X": X,
            "X_intact": X_intact,
            "missing_mask": missing_mask,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def assemble_input_for_validating(self, data) -> dict:
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
        indices, X, missing_mask = data

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }
        return inputs

    def assemble_input_for_testing(self, data) -> dict:
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
        return self.assemble_input_for_validating(data)

    def impute(self, X, file_type="h5py"):
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        X : array-like or str,
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type : str, default = "h5py",
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (time steps), n_features],
            Imputed data.
        """
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(X, file_type)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = {"X": data[1], "missing_mask": data[2]}
                imputed_data, _ = self.model.impute(inputs)
                imputation_collector.append(imputed_data)

        imputation_collector = torch.cat(imputation_collector)
        return imputation_collector.cpu().detach().numpy()
