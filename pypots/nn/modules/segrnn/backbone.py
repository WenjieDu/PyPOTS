"""

"""

# Created by Shengsheng Lin

import torch
import torch.nn as nn


class BackboneSegRNN(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        seg_len: int = 24,
        d_model: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()

        if n_steps % seg_len:
            raise ValueError("The argument seg_len in SegRNN need to be divisible by the sequence length n_steps.")
        if n_pred_steps % seg_len:
            raise ValueError(
                "The argument seg_len in SegRNN need to be divisible by the prediction sequence length n_pred_steps."
            )

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.seg_len = seg_len
        self.d_model = d_model
        self.dropout = dropout

        self.seg_num_x = self.n_steps // self.seg_len
        self.seg_num_y = self.n_pred_steps // self.seg_len
        self.valueEmbedding = nn.Sequential(nn.Linear(self.seg_len, self.d_model), nn.ReLU())
        self.rnn = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.n_features, self.d_model // 2))
        self.predict = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.seg_len))

    def forward(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n m:seg_num
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)  # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn(x)  # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = (
            torch.cat(
                [
                    self.pos_emb.unsqueeze(0).repeat(self.n_features, 1, 1),
                    self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1),
                ],
                dim=-1,
            )
            .view(-1, 1, self.d_model)
            .repeat(batch_size, 1, 1)
        )

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))  # bcm,1,d  1,bcm,d

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.n_features, self.n_pred_steps)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_last

        return y
