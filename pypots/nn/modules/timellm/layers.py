"""

"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: BSD-3-Clause


from math import sqrt

import torch
import torch.nn as nn


class ReprogrammingLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int = None,
        d_llm: int = None,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        d_k = d_k or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_k * n_heads)
        self.key_projection = nn.Linear(d_llm, d_k * n_heads)
        self.value_projection = nn.Linear(d_llm, d_k * n_heads)
        self.out_projection = nn.Linear(d_k * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1.0 / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
