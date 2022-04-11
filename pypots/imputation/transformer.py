"""
PyTorch Transformer model for the time-series imputation task.
Some part of the code is from https://github.com/WenjieDu/SAITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pypots.data.base import Dataset4MIT
from pypots.imputation.base import BaseImputer
from pypots.utils.metrics import cal_mae


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, attn_dropout=0.1,
                 diagonal_attention_mask=False, device=None):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = diagonal_attention_mask
        self.device = device
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(self.device)
        else:
            mask_time = None

        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=mask_time)
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class _TransformerEncoder(nn.Module):
    def __init__(self, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 ORT_weight=1, MIT_weight=1, device=None):
        super(_TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.device = device

        self.layer_stack_for_first_block = nn.ModuleList([
            EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
                         False, device)
            for _ in range(n_layers)
        ])
        self.layer_stack_for_second_block = nn.ModuleList([
            EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
                         False, device)
            for _ in range(n_layers)
        ])

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        input_X = torch.cat([X, masks], dim=2)
        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, _ = encoder_layer(enc_output)

        learned_presentation = self.reduce_dim(enc_output)
        imputed_data = masks * X + (1 - masks) * learned_presentation  # replace non-missing part with original data
        return imputed_data, learned_presentation

    def forward(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        imputed_data, learned_presentation = self.impute(inputs)
        reconstruction_loss = cal_mae(learned_presentation, X, masks)

        # have to cal imputation loss in the val stage; no need to cal imputation loss here in the tests stage
        imputation_loss = cal_mae(learned_presentation, inputs['X_holdout'], inputs['indicating_mask'])

        loss = self.ORT_weight * reconstruction_loss + self.MIT_weight * imputation_loss

        return {
            'imputed_data': imputed_data,
            'reconstruction_loss': reconstruction_loss, 'imputation_loss': imputation_loss,
            'loss': loss
        }


class Transformer(BaseImputer):
    def __init__(self,
                 seq_len,
                 n_features,
                 n_layers,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout,
                 learning_rate=1e-3,
                 epochs=100,
                 patience=10,
                 batch_size=32,
                 weight_decay=1e-5,
                 ORT_weight=1,
                 MIT_weight=1,
                 device=None):
        super(Transformer, self).__init__()

        # model hype-parameters
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # training hype-parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = learning_rate
        self.weight_decay = weight_decay

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = None
        self.optimizer = None
        self.data_loader = None
        self.best_loss = float('inf')
        self.best_model_dict = None

    def fit(self, X):
        self.model = _TransformerEncoder(self.n_layers, self.seq_len, self.n_features, self.d_model, self.d_inner,
                                         self.n_head, self.d_k, self.d_v, self.dropout,
                                         self.ORT_weight, self.MIT_weight, self.device)
        self.model = self.model.to(self.device)
        training_set = Dataset4MIT(X)
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        self._train_model(training_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.
        return self

    def _train_model(self, training_loader):
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float('inf')
        self.best_model_dict = None

        for epoch in range(self.epochs):
            loss_collector = []
            for idx, data in enumerate(training_loader):
                # fetch data
                indices, X_intact, X, missing_mask, indicating_mask = \
                    map(lambda x: x.to(self.device), data)

                inputs = {
                    'X': X,
                    'X_intact': X_intact,
                    'missing_mask': missing_mask,
                    'indicating_mask': indicating_mask
                }

                self.optimizer.zero_grad()
                results = self.model.forward(inputs)
                results['loss'].backward()
                self.optimizer.step()
                loss_collector.append(results['loss'].item())

            epoch_mean_loss = np.mean(loss_collector)  # mean loss of the current epoch
            print(f'epoch {epoch}: training loss {epoch_mean_loss:.4f} ')

            if epoch_mean_loss < self.best_loss:
                self.best_loss = epoch_mean_loss
                self.best_model_dict = self.model.state_dict()
            else:
                self.patience -= 1
                if self.patience == 0:
                    break

    def impute(self, X):
        test_set = Dataset4MIT(X)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                indices, X_intact, X, missing_mask, indicating_mask = \
                    map(lambda x: x.to(self.device), data)
                # assemble input data
                inputs = {
                    'X': X,
                    'X_intact': X_intact,
                    'missing_mask': missing_mask,
                    'indicating_mask': indicating_mask
                }
                imputed_data, _ = self.model.impute(inputs)
                imputation_collector.append(imputed_data)

        imputation_collector = torch.cat(imputation_collector)
        return imputation_collector.cpu().detach().numpy()
