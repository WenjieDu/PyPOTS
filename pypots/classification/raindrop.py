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


import math
from typing import Union, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import PairTensor, Adj, OptTensor
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_sparse import SparseTensor

from pypots.classification.base import BaseNNClassifier
from pypots.data.dataset_for_grud import DatasetForGRUD


class PositionalEncodingTF(nn.Module):
    """ Generate positional encoding according to time information.
    """

    def __init__(self, d_pe, max_len=500):
        super().__init__()
        assert d_pe % 2 == 0, 'd_pe should be even, otherwise the output dims will be not equal to d_pe'
        self.max_len = max_len
        self._num_timescales = d_pe // 2

    def forward(self, time_vectors):
        """ Generate positional encoding.

        Parameters
        ----------
        time_vectors : tensor,
            Tensor embeds time information.

        Returns
        -------
        pe : tensor,
            Positional encoding with d_pe dims.
        """
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = time_vectors.unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)
        return pe


class ObservationPropagation(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int,
                 n_nodes: int, ob_dim: int, heads: int = 1, concat: bool = True,
                 beta: bool = False, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.weight = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.bias = Parameter(torch.Tensor(heads * out_channels))

        self.n_nodes = n_nodes
        self.nodewise_weights = Parameter(torch.Tensor(self.n_nodes, heads * out_channels))

        self.increase_dim = Linear(in_channels[1], heads * out_channels * 8)
        self.map_weights = Parameter(torch.Tensor(self.n_nodes, heads * 16))

        self.ob_dim = ob_dim
        self.index = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()
        glorot(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        glorot(self.nodewise_weights)
        glorot(self.map_weights)
        self.increase_dim.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], p_t: Tensor, edge_index: Adj, edge_weights=None, use_beta=False,
                edge_attr: OptTensor = None, return_attention_weights=None):

        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        """Here, the edge_attr is not edge weights, but edge features!
        If we want to the calculation contains edge weights, change the calculation of alpha"""

        self.edge_index = edge_index
        self.p_t = p_t
        self.use_beta = use_beta

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weights=edge_weights, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None
        edge_index = self.edge_index

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message_selfattention(self, x_i: Tensor, x_j: Tensor, edge_weights: Tensor, edge_attr: OptTensor,
                              index: Tensor, ptr: OptTensor,
                              size_i: Optional[int]) -> Tensor:
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        if edge_weights is not None:
            alpha = edge_weights.unsqueeze(-1)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        out *= alpha.view(-1, self.heads, 1)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        use_beta = self.use_beta
        if use_beta:
            n_step = self.p_t.shape[0]
            n_edges = x_i.shape[0]

            h_W = self.increase_dim(x_i).view(-1, n_step, 32)
            w_v = self.map_weights[self.edge_index[1]].unsqueeze(1)

            p_emb = self.p_t.unsqueeze(0)

            aa = torch.cat([w_v.repeat(1, n_step, 1, ), p_emb.repeat(n_edges, 1, 1)], dim=-1)
            beta = torch.mean(h_W * aa, dim=-1)

        if edge_weights is not None:
            if use_beta:
                gamma = beta * (edge_weights.unsqueeze(-1))
                gamma = torch.repeat_interleave(gamma, self.ob_dim, dim=-1)

                # edge prune, prune out half of edges
                all_edge_weights = torch.mean(gamma, dim=1)
                K = int(gamma.shape[0] * 0.5)
                index_top_edges = torch.argsort(all_edge_weights, descending=True)[:K]
                gamma = gamma[index_top_edges]
                self.edge_index = self.edge_index[:, index_top_edges]
                index = self.edge_index[0]
                x_i = x_i[index_top_edges]
            else:
                gamma = edge_weights.unsqueeze(-1)

        self.index = index
        if use_beta:
            self._alpha = torch.mean(gamma, dim=-1)
        else:
            self._alpha = gamma

        gamma = softmax(gamma, index, ptr, size_i)
        gamma = F.dropout(gamma, p=self.dropout, training=self.training)

        decompose = False
        if not decompose:
            out = F.relu(self.lin_value(x_i)).view(-1, self.heads, self.out_channels)
        else:
            source_nodes = self.edge_index[0]
            target_nodes = self.edge_index[1]
            w1 = self.nodewise_weights[source_nodes].unsqueeze(-1)
            w2 = self.nodewise_weights[target_nodes].unsqueeze(1)
            out = torch.bmm(x_i.view(-1, self.heads, self.out_channels), torch.bmm(w1, w2))
        if use_beta:
            out = out * gamma.view(-1, self.heads, out.shape[-1])
        else:
            out = out * gamma.view(-1, self.heads, 1)
        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        index = self.index
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels,
                                             self.heads)


class _Raindrop(nn.Module):
    def __init__(self, n_layers, n_features, d_model, d_inner, n_heads, n_classes, dropout=0.3, max_len=215, d_static=9,
                 aggregation='mean', sensor_wise_mask=False, static=False, device=None):
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
        assert d_model % n_features == 0, 'd_model must be divisible by n_features'
        self.d_ob = int(d_model / n_features)
        self.encoder = nn.Linear(n_features * self.d_ob, n_features * self.d_ob)
        d_pe = 16
        self.pos_encoder = PositionalEncodingTF(d_pe, max_len)
        if self.sensor_wise_mask:
            dim_check = n_features * (self.d_ob + d_pe)
            assert dim_check % n_heads == 0, 'dim_check must be divisible by n_heads'
            encoder_layers = TransformerEncoderLayer(n_features * (self.d_ob + d_pe), n_heads, d_inner, dropout)
        else:
            dim_check = d_model + d_pe
            assert dim_check % n_heads == 0, 'dim_check must be divisible by n_heads'
            encoder_layers = TransformerEncoderLayer(d_model + d_pe, n_heads, d_inner, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.adj = torch.ones([self.n_features, self.n_features], device=self.device)

        self.R_u = Parameter(torch.Tensor(1, self.n_features * self.d_ob))

        self.ob_propagation = ObservationPropagation(in_channels=max_len * self.d_ob, out_channels=max_len * self.d_ob,
                                                     heads=1, n_nodes=n_features, ob_dim=self.d_ob)
        self.ob_propagation_layer2 = ObservationPropagation(in_channels=max_len * self.d_ob,
                                                            out_channels=max_len * self.d_ob, heads=1,
                                                            n_nodes=n_features, ob_dim=self.d_ob)
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

    def classify(self, inputs):
        """ Forward processing of BRITS.

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
        src = inputs['X']
        static = inputs['static']
        times = inputs['timestamps']
        lengths = inputs['lengths']
        missing_mask = inputs['missing_mask']

        max_len, batch_size = src.shape[0], src.shape[1]

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src * self.R_u)
        pe = self.pos_encoder(times)
        if static is not None:
            emb = self.emb(static)

        h = self.dropout(h)

        mask = torch.arange(max_len)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1)

        x = h

        adj = self.global_structure
        adj[torch.eye(self.n_features).byte()] = 1

        edge_index = torch.nonzero(adj).T
        edge_weights = adj[edge_index[0], edge_index[1]]

        batch_size = src.shape[1]
        n_step = src.shape[0]
        output = torch.zeros([n_step, batch_size, self.n_features * self.d_ob], device=self.device)

        alpha_all = torch.zeros([edge_index.shape[1], batch_size], device=self.device)

        # iterate on each sample
        for unit in range(0, batch_size):
            step_data = x[:, unit, :]
            p_t = pe[:, unit, :]

            step_data = step_data.reshape([n_step, self.n_features, self.d_ob]).permute(1, 0, 2)
            step_data = step_data.reshape(self.n_features, n_step * self.d_ob)

            step_data, attention_weights = self.ob_propagation(step_data, p_t=p_t, edge_index=edge_index,
                                                               edge_weights=edge_weights,
                                                               use_beta=False, edge_attr=None,
                                                               return_attention_weights=True)

            edge_index_layer2 = attention_weights[0]
            edge_weights_layer2 = attention_weights[1].squeeze(-1)

            step_data, attention_weights = self.ob_propagation_layer2(step_data, p_t=p_t, edge_index=edge_index_layer2,
                                                                      edge_weights=edge_weights_layer2,
                                                                      use_beta=False, edge_attr=None,
                                                                      return_attention_weights=True)

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

        lengths2 = lengths.unsqueeze(1)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        if sensor_wise_mask:
            output = torch.zeros([batch_size, self.n_features, self.d_ob + 16], device=self.device)
            extended_missing_mask = missing_mask.view(-1, batch_size, self.n_features)
            for se in range(self.n_features):
                r_out = r_out.view(-1, batch_size, self.n_features, (self.d_ob + 16))
                out = r_out[:, :, se, :]
                l_ = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)  # length
                out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (l_ + 1)
                output[:, se, :] = out_sensor
            output = output.view([-1, self.n_features * (self.d_ob + 16)])
        elif self.aggregation == 'mean':
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
        classification_loss = F.nll_loss(torch.log(prediction), inputs['label'])

        results = {
            'prediction': prediction,
            'loss': classification_loss
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

    def __init__(self,
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
                 learning_rate=1e-3,
                 epochs=100,
                 patience=10,
                 batch_size=32,
                 weight_decay=1e-5,
                 device=None):
        super().__init__(n_classes, learning_rate, epochs, patience, batch_size,
                         weight_decay, device)

        self.n_features = n_features
        self.n_steps = max_len
        self.model = _Raindrop(n_layers, n_features, d_model, d_inner, n_heads, n_classes, dropout, max_len, d_static,
                               aggregation, sensor_wise_mask, static=static, device=self.device)
        self.model = self.model.to(self.device)
        self._print_model_size()

    def fit(self, train_X, train_y, val_X=None, val_y=None):
        """ Fit the model on the given training data.

        Parameters
        ----------
        train_X : array, shape [n_samples, sequence length (time steps), n_features],
            Time-series vectors.
        train_y : array,
            Classification labels.

        Returns
        -------
        self : object,
            Trained model.
        """
        train_X, train_y = self.check_input(self.n_steps, self.n_features, train_X, train_y)
        val_X, val_y = self.check_input(self.n_steps, self.n_features, val_X, val_y)

        training_set = DatasetForGRUD(train_X, train_y)
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)

        if val_X is None:
            self._train_model(training_loader)
        else:
            val_set = DatasetForGRUD(val_X, val_y)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            self._train_model(training_loader, val_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.
        return self

    def assemble_input_data(self, data):
        """ Assemble the input data into a dictionary.

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
        indices, X, X_filledLOCF, missing_mask, deltas, empirical_mean, label = data

        bz, n_steps, n_features = X.shape
        lengths = torch.tensor([n_steps] * bz, dtype=torch.float)
        times = torch.tensor(range(n_steps), dtype=torch.float).repeat(bz, 1)

        X = X.permute(1, 0, 2)
        missing_mask = missing_mask.permute(1, 0, 2)
        times = times.permute(1, 0)

        inputs = {
            'X': X,
            'static': None,
            'timestamps': times,
            'lengths': lengths,
            'missing_mask': missing_mask,
            'label': label
        }
        return inputs

    def classify(self, X):
        X = self.check_input(self.n_steps, self.n_features, X)
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGRUD(X)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        prediction_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                # cannot use input_data_processing, cause here has no label
                indices, X, X_filledLOCF, missing_mask, deltas, empirical_mean = data
                # assemble input data

                bz, n_steps, n_features = X.shape
                lengths = torch.tensor([n_steps] * bz, dtype=torch.float)
                times = torch.tensor(range(n_steps), dtype=torch.float).repeat(bz, 1)

                X = X.permute(1, 0, 2)
                missing_mask = missing_mask.permute(1, 0, 2)
                times = times.permute(1, 0)

                inputs = {
                    'X': X,
                    'static': None,
                    'timestamps': times,
                    'lengths': lengths,
                    'missing_mask': missing_mask,
                }

                prediction = self.model.classify(inputs)
                prediction_collector.append(prediction)

        predictions = torch.cat(prediction_collector)
        return predictions.cpu().detach().numpy()
