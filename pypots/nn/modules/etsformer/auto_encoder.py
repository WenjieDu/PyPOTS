"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn


class ETSformerEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, res, level, attn_mask=None):
        growths = []
        seasons = []
        for layer in self.layers:
            res, level, growth, season = layer(res, level, attn_mask=attn_mask)
            growths.append(growth)
            seasons.append(season)

        return level, growths, seasons


class ETSformerDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.d_model = layers[0].d_model
        self.d_out = layers[0].d_out
        self.pred_len = layers[0].pred_len
        self.n_head = layers[0].n_heads

        self.layers = nn.ModuleList(layers)
        self.pred = nn.Linear(self.d_model, self.d_out)

    def forward(self, growths, seasons):
        growth_repr = []
        season_repr = []

        for idx, layer in enumerate(self.layers):
            growth_horizon, season_horizon = layer(growths[idx], seasons[idx])
            growth_repr.append(growth_horizon)
            season_repr.append(season_horizon)
        growth_repr = sum(growth_repr)
        season_repr = sum(season_repr)
        return self.pred(growth_repr), self.pred(season_repr)
