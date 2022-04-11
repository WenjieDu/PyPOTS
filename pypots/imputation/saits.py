"""
PyTorch SAITS model for the time-series imputation task.
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
from pypots.imputation.transformer import EncoderLayer, PositionalEncoding
from pypots.utils.metrics import cal_mae


class _SAITS(nn.Module):
    def __init__(self, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 diagonal_attention_mask=True, ORT_weight=1, MIT_weight=1, device=None):
        super(_SAITS, self).__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.device = device

        self.layer_stack_for_first_block = nn.ModuleList([
            EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask, device)
            for _ in range(n_layers)
        ])
        self.layer_stack_for_second_block = nn.ModuleList([
            EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask, device)
            for _ in range(n_layers)
        ])

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
        X, masks = inputs['X'], inputs['missing_mask']
        # first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2)
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(self.position_enc(input_X_for_first))  # namely, term e in the math equation
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        input_X_for_second = torch.cat([X_prime, masks], dim=2)
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(input_X_for_second)  # namely term alpha in math algo
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze()  # namely term A_hat in Eq.
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
        X_c = masks * X + (1 - masks) * X_tilde_3  # replace non-missing part with original data
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)

        reconstruction_loss += cal_mae(X_tilde_1, X, masks)
        reconstruction_loss += cal_mae(X_tilde_2, X, masks)
        final_reconstruction_MAE = cal_mae(X_tilde_3, X, masks)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        # have to cal imputation loss in the val stage; no need to cal imputation loss here in the tests stage
        imputation_loss = cal_mae(X_tilde_3, inputs['X_intact'], inputs['indicating_mask'])

        loss = self.ORT_weight * reconstruction_loss + self.MIT_weight * imputation_loss

        return {
            'imputed_data': imputed_data,
            'reconstruction_loss': reconstruction_loss, 'imputation_loss': imputation_loss,
            'loss': loss
        }


class SAITS(BaseImputer):
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
                 diagonal_attention_mask=True,
                 learning_rate=1e-3,
                 epochs=100,
                 patience=10,
                 batch_size=32,
                 weight_decay=1e-5,
                 ORT_weight=1,
                 MIT_weight=1,
                 device=None):
        super(SAITS, self).__init__()

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
        self.diagonal_attention_mask = diagonal_attention_mask
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
        self.model = _SAITS(self.n_layers, self.seq_len, self.n_features, self.d_model, self.d_inner, self.n_head,
                            self.d_k, self.d_v, self.dropout, self.diagonal_attention_mask,
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
