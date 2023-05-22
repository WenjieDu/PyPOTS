"""
The implementation of Raindrop for the partially-observed time-series classification task.

Refer to the paper "Zhang, X., Zeman, M., Tsiligkaridis, T., & Zitnik, M. (2022).
Graph-Guided Network for Irregularly Sampled Multivariate Time Series. ICLR 2022."

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

from ...classification.base import BaseNNClassifier
from ...classification.grud.data import DatasetForGRUD
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger

try:
    from .modules import PositionalEncoding, ObservationPropagation
    from torch_geometric.nn.inits import glorot
except ImportError as e:
    logger.error(
        f"{e}\n"
        "Note torch_geometric is missing, "
        "please install it with 'pip install torch_geometric' or 'conda install -c pyg pyg'"
    )
except NameError as e:
    logger.error(
        f"{e}\n"
        "Note torch_geometric is missing, "
        "please install it with 'pip install torch_geometric' or 'conda install -c pyg pyg'"
    )


class _Raindrop(nn.Module):
    def __init__(
        self,
        n_features,
        n_layers,
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
            self.static_emb = nn.Linear(d_static, n_features)
        else:
            self.static_emb = None
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
            self.static_emb.weight.data.uniform_(-init_range, init_range)
        glorot(self.R_u)

    def classify(self, inputs: dict) -> torch.Tensor:
        """Forward processing of BRITS.

        Parameters
        ----------
        inputs : dict,
            The input data.

        Returns
        -------
        prediction : torch.Tensor
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
            emb = self.static_emb(static)

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

    def forward(self, inputs, training=True):
        classification_pred = self.classify(inputs)
        if not training:
            # if not in training mode, return the classification result only
            return {"classification_pred": classification_pred}

        classification_loss = F.nll_loss(
            torch.log(classification_pred), inputs["label"]
        )

        results = {
            "prediction": classification_pred,
            "loss": classification_loss
            # 'distance': distance,
        }

        return results


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
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    Attributes
    ----------
    model : :class:`torch.nn.Module`
        The underlying Raindrop model.

    optimizer : :class:`pypots.optim.Optimizer`
        The optimizer for model training.

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
        patience: int = None,
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

    def _assemble_input_for_training(self, data: dict) -> dict:
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
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: dict) -> dict:
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
                results = self.model.forward(inputs, training=False)
                prediction = results["classification_pred"]
                prediction_collector.append(prediction)

        predictions = torch.cat(prediction_collector)
        return predictions.cpu().detach().numpy()
