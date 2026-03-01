"""
The core wrapper assembles the submodules of UniTS imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.units import (
    PatchEmbedding,
    LearnablePositionalEmbedding,
    DynamicLinear,
    BasicBlock,
    ForecastHead,
)


class _UniTS(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        patch_len: int,
        stride: int,
        dropout: float,
        prompt_num: int,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.prompt_num = prompt_num
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        # Prompt and mask tokens (per-dataset, here a single dataset is used)
        self.prompt_token = nn.Parameter(torch.zeros(1, n_features, prompt_num, d_model), requires_grad=True)
        nn.init.normal_(self.prompt_token, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, n_features, 1, d_model), requires_grad=True)

        # Input processing
        self.patch_embeddings = PatchEmbedding(d_model, patch_len, stride, dropout)
        self.position_embedding = LearnablePositionalEmbedding(d_model)
        self.prompt2forecast = DynamicLinear(128, 128, fixed_in=prompt_num)

        # Backbone blocks
        self.blocks = nn.ModuleList(
            [
                BasicBlock(
                    dim=d_model,
                    num_heads=n_heads,
                    qkv_bias=False,
                    qk_norm=False,
                    mlp_ratio=8.0,
                    proj_drop=dropout,
                    attn_drop=0.0,
                    drop_path=0.0,
                    init_values=None,
                    prefix_token_length=prompt_num,
                )
                for _ in range(n_layers)
            ]
        )

        # Output head
        self.forecast_head = ForecastHead(
            d_model,
            patch_len,
            stride,
            stride,
            prefix_token_length=prompt_num,
            head_dropout=dropout,
        )

    def tokenize(self, x, mask=None):
        """Normalize and convert to patch embeddings."""
        # Normalization (instance normalization)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(
                torch.sum(x * x, dim=1) / torch.sum(mask == 1, dim=1).clamp(min=1) + 1e-5
            )
            stdev = stdev.unsqueeze(dim=1)
        else:
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # Permute to (B, n_vars, seq_len) for patching
        x = x.permute(0, 2, 1)
        remainder = x.shape[2] % self.patch_len
        if remainder != 0:
            padding = self.patch_len - remainder
            x = F.pad(x, (0, padding))
        else:
            padding = 0
        x, n_vars = self.patch_embeddings(x)
        return x, means, stdev, n_vars, padding

    def mark2token(self, x_mark):
        """Convert mask from timestep space to token space."""
        x_mark = x_mark.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_mark = x_mark.mean(dim=-1)
        x_mark = (x_mark > 0).float()
        return x_mark

    def backbone(self, x, prefix_len, seq_len):
        """Pass through all transformer blocks."""
        for block in self.blocks:
            x = block(x, prefix_seq_len=prefix_len + seq_len, attn_mask=None)
        return x

    def imputation(self, x, mask):
        """Full imputation forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (B, L, D), where missing values are zero.

        mask : torch.Tensor
            Mask of shape (B, L, D), where 1=observed, 0=missing.

        Returns
        -------
        torch.Tensor
            Reconstruction of shape (B, L, D).
        """
        seq_len = x.shape[1]
        x, means, stdev, n_vars, padding = self.tokenize(x, mask)

        # Reshape from (B*n_vars, n_tokens, d) to (B, n_vars, n_tokens, d)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))

        # Convert mask from (B, L, D) -> (B, D, L) -> token space
        # In imputation: mask=1 is observed, mask=0 is missing
        # For filling: we want to identify missing tokens
        mask_token_mask = 1 - mask  # 1 where missing
        mask_token_mask = mask_token_mask.permute(0, 2, 1)  # (B, D, L)
        mask_token_mask = self.mark2token(mask_token_mask)  # (B, D, n_tokens)
        mask_repeat = mask_token_mask.unsqueeze(dim=-1)  # (B, D, n_tokens, 1)
        mask_repeat = mask_repeat.repeat(1, 1, 1, x.shape[-1])  # (B, D, n_tokens, d_model)

        # Fill missing positions with mask token
        mask_token = self.mask_token
        x = x * (1 - mask_repeat) + mask_token * mask_repeat

        # Prepare prompt
        this_prompt = self.prompt_token.repeat(x.shape[0], 1, 1, 1)

        # Apply dynamic linear for masked positions
        init_full_input = torch.cat((this_prompt, x), dim=-2)
        init_mask_prompt = self.prompt2forecast(
            init_full_input.transpose(-1, -2), x.shape[2]
        ).transpose(-1, -2)
        # Keep observed tokens, replace missing with learned representation
        x = x * (1 - mask_repeat) + init_mask_prompt * mask_repeat

        # Positional embedding
        x = x + self.position_embedding(x)

        # Prepend prompt tokens
        x = torch.cat((this_prompt, x), dim=2)

        # Run backbone
        seq_token_len = x.shape[-2] - self.prompt_token.shape[2]
        x = self.backbone(x, self.prompt_token.shape[2], seq_token_len)

        # Decode
        x = self.forecast_head(x, seq_len + padding, seq_token_len)
        x = x[:, :seq_len]

        # De-normalization
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        return x

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        reconstruction = self.imputation(X, missing_mask)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                loss = self.training_loss(reconstruction, X, missing_mask)
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
