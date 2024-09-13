"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import math
from typing import Tuple, Any, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.nn import init
from torch.nn.parameter import Parameter

try:
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.nn.inits import glorot
    from torch_geometric.typing import PairTensor, Adj, OptTensor
    from torch_geometric.utils import softmax
    from torch_scatter import scatter
    from torch_sparse import SparseTensor
except ImportError:
    # Modules here only for Raindrop model, and torch_geometric import errors are caught in BackboneRaindrop.
    # Hence, we can pass them here.
    pass


class PositionalEncoding(nn.Module):
    """Generate positional encoding according to time information."""

    def __init__(self, d_pe: int, max_len: int = 500):
        super().__init__()
        assert d_pe % 2 == 0, "d_pe should be even, otherwise the output dims will be not equal to d_pe"
        self.max_len = max_len
        self._num_timescales = d_pe // 2

    def forward(self, time_vectors: torch.Tensor) -> torch.Tensor:
        """Generate positional encoding.

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
        scaled_time = times / torch.from_numpy(timescales[None, None, :]).to(time_vectors.device)
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)
        return pe


class ObservationPropagation(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        n_nodes: int,
        ob_dim: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
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
            self.lin_edge = self.register_parameter("lin_edge", None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)

        self.weight = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.bias = Parameter(torch.Tensor(heads * out_channels))

        self.n_nodes = n_nodes
        self.nodewise_weights = Parameter(torch.Tensor(self.n_nodes, heads * out_channels))

        self.increase_dim = Linear(in_channels[1], heads * out_channels * 8)
        self.map_weights = Parameter(torch.Tensor(self.n_nodes, heads * 16))

        self.ob_dim = ob_dim
        self.index = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge._reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta._reset_parameters()
        glorot(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        glorot(self.nodewise_weights)
        glorot(self.map_weights)
        self.increase_dim.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        p_t: Tensor,
        edge_index: Adj,
        edge_weights=None,
        use_beta=False,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ) -> Tuple[torch.Tensor, Any]:
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
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message_selfattention(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_weights: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
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

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_weights: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        use_beta = self.use_beta
        if use_beta:
            n_step = self.p_t.shape[0]
            n_edges = x_i.shape[0]

            h_W = self.increase_dim(x_i).view(-1, n_step, 32)
            w_v = self.map_weights[self.edge_index[1]].unsqueeze(1)

            p_emb = self.p_t.unsqueeze(0)

            aa = torch.cat(
                [
                    w_v.repeat(
                        1,
                        n_step,
                        1,
                    ),
                    p_emb.repeat(n_edges, 1, 1),
                ],
                dim=-1,
            )
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

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        index = self.index
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)
