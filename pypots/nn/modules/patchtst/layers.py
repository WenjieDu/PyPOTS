"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ....nn.modules.transformer.embedding import PositionalEncoding


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model,
        patch_size,
        patch_stride,
        padding,
        dropout,
        positional_embedding=True,
    ):
        super().__init__()
        # patching
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        # input projection, project the feature vectors into a vector space with d_model dimensions
        self.value_embedding = nn.Linear(patch_size, d_model, bias=False)
        # positional embedding
        if positional_embedding:
            self.positional_embedding = PositionalEncoding(d_model)
        else:
            self.positional_embedding = None
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # apply patching
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        # input encoding
        x = self.value_embedding(x)
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        x = self.dropout(x)
        return x


class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x):
        # return sigmoid_range(x, self.low, self.high)
        return torch.sigmoid(x) * (self.high - self.low) + self.low


class RegressionHead(nn.Module):
    def __init__(self, n_features, d_model, d_output, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_features * d_model, d_output)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:, :, :, -1]  # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)  # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: bs x output_dim
        if self.y_range:
            y = SigmoidRange(*self.y_range)(y)
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_features, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_features * d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:, :, :, -1]  # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)  # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: bs x n_classes
        return y


class PredictionHead(nn.Module):
    def __init__(
        self,
        d_model,
        n_patches,
        n_steps_forecast,
        head_dropout=0,
        individual=False,
        n_features=0,
    ):
        super().__init__()

        head_dim = d_model * n_patches
        self.individual = individual
        self.n_features = n_features

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_features):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, n_steps_forecast))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, n_steps_forecast)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_features):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * num_patch]
                z = self.linears[i](z)  # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)  # x: [bs x nvars x (d_model * num_patch)]
            x = self.dropout(x)
            x = self.linear(x)  # x: [bs x nvars x forecast_len]
        return x.transpose(2, 1)  # [bs x forecast_len x nvars]


class FlattenHead(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        n_features,
        head_dropout=0,
        individual=False,
    ):
        super().__init__()

        self.individual = individual
        self.n_features = n_features

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_features):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(d_input, d_output))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(d_input, d_output)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_features):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
