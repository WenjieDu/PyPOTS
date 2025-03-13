"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import warnings
from argparse import Namespace
from dataclasses import dataclass

import numpy.typing as npt
import torch
from torch import nn

from .masking import Masking
from ....nn.modules.transformer.embedding import PositionalEncoding


@dataclass
class TASKS:
    PRETRAINING: str = "pre-training"
    LONG_HORIZON_FORECASTING: str = "long_term_forecasting"
    SHORT_HORIZON_FORECASTING: str = "short_term_forecasting"
    CLASSIFICATION: str = "classification"
    IMPUTATION: str = "imputation"
    ANOMALY_DETECTION: str = "anomaly-detection"
    EMBED: str = "embed"


@dataclass
class TimeseriesOutputs:
    forecast: npt.NDArray = None
    anomaly_scores: npt.NDArray = None
    labels: int = None
    input_mask: npt.NDArray = None
    pretrain_mask: npt.NDArray = None
    reconstruction: npt.NDArray = None
    embeddings: npt.NDArray = None
    metadata: dict = None
    illegal_output: bool = False


class NamespaceWithDefaults(Namespace):
    @classmethod
    def from_namespace(cls, namespace):
        new_instance = cls()
        for attr in dir(namespace):
            if not attr.startswith("__"):
                setattr(new_instance, attr, getattr(namespace, attr))
        return new_instance

    def getattr(self, key, default=None):
        return getattr(self, key, default)


def get_anomaly_criterion(anomaly_criterion: str = "mse"):
    if anomaly_criterion == "mse":
        return torch.nn.MSELoss(reduction="none")
    elif anomaly_criterion == "mae":
        return torch.nn.L1Loss(reduction="none")
    else:
        raise ValueError(f"Anomaly criterion {anomaly_criterion} not supported.")


def get_huggingface_model_dimensions(model_name: str = "flan-t5-base"):
    from transformers import T5Config

    config = T5Config.from_pretrained(model_name)
    return config.d_model


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


class Patching(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if self.patch_stride != self.patch_size:
            warnings.warn(
                "Stride and patch length are not equal. \
                          This may lead to unexpected behavior."
            )

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        # x : [batch_size x n_channels x num_patch x patch_size]
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        seq_len: int = 512,
        patch_size: int = 8,
        patch_stride: int = 8,
        dropout: int = 0.1,
        add_positional_embedding: bool = False,
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        # Patching
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.add_positional_embedding = add_positional_embedding

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_size, d_model, bias=value_embedding_bias)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        # nn.init.trunc_normal_(self.mask_embedding, mean=0.0, std=.02)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
            if value_embedding_bias:
                self.value_embedding.bias.data.zero_()
            # torch.nn.init.orthogonal_(self.mask_embedding, gain=orth_gain) # Fails

        # Positional embedding
        if self.add_positional_embedding:
            self.position_embedding = PositionalEncoding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Input:
            x : [batch_size x n_channels x n_patches x patch_size]
            mask : [batch_size x seq_len]
        Output:
            x : [batch_size x n_channels x n_patches x d_model]
        """

        mask = Masking.convert_seq_to_patch_view(mask, patch_size=self.patch_size).unsqueeze(-1)
        # mask : [batch_size x n_patches x 1]
        n_channels = x.shape[1]
        mask = mask.repeat_interleave(self.d_model, dim=-1).unsqueeze(1).repeat(1, n_channels, 1, 1)
        # mask : [batch_size x n_channels x n_patches x d_model]

        # Input encoding
        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
        if self.add_positional_embedding:
            x = x + self.position_embedding(x, dim=2, return_only_pos=True)

        return self.dropout(x)


class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_size: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_size)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x seq_len], where seq_len = n_patches * patch_size
        """
        # x = x.transpose(2, 3)                 # [batch_size x n_channels x n_patches x d_model]
        x = self.linear(self.dropout(x))  # [batch_size x n_channels x n_patches x patch_size]
        x = x.flatten(start_dim=2, end_dim=3)  # [batch_size x n_patches x seq_len]
        return x


class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: int = 0.1,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_channels * d_model, n_classes)

    def forward(self, x, input_mask: torch.Tensor = None):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_classes]
        """
        x = x.nanmean(dim=-1).squeeze()  # x: batch_size x n_channels x n_patches x d_model
        x = self.flatten(x)  # x: batch_size x n_channels * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: batch_size x n_classes
        return y


class ForecastingHead(nn.Module):
    def __init__(self, head_nf: int = 768 * 64, forecast_horizon: int = 96, head_dropout: int = 0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x forecast_horizon]
        """
        x = self.flatten(x)  # x: batch_size x n_channels x n_patches x d_model
        x = self.linear(x)  # x: batch_size x n_channels x n_patches*d_model
        x = self.dropout(x)  # x: batch_size x n_channels x forecast_horizon
        return x


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str = "norm", mask: torch.Tensor = None):
        """
        :param x: input tensor of shape (batch_size, n_channels, seq_len)
        :param mode: 'norm' or 'denorm'
        :param mask: input mask of shape (batch_size, seq_len)
        :return: RevIN transformed tensor
        """
        if mode == "norm":
            self._get_statistics(x, mask=mask)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def _get_statistics(self, x, mask=None):
        """
        x    : batch_size x n_channels x seq_len
        mask : batch_size x seq_len
        """
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]))
        n_channels = x.shape[1]
        mask = mask.unsqueeze(1).repeat(1, n_channels, 1).bool()
        # Set masked positions to NaN, and unmasked positions are taken from x
        masked_x = torch.where(mask, x, torch.nan)
        self.mean = torch.nanmean(masked_x, dim=-1, keepdim=True).detach()
        self.stdev = nanstd(masked_x, dim=-1, keepdim=True).detach() + self.eps
        # self.stdev = torch.sqrt(
        #     torch.var(masked_x, dim=-1, keepdim=True) + self.eps).get_data().detach()
        # NOTE: By default not bessel correction

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
