"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from ..transformer.embedding import DataEmbedding


class BackboneGPT4TS(nn.Module):
    def __init__(
        self,
        task_name,
        n_steps,
        n_features,
        n_pred_steps,
        n_pred_features,
        n_layers,
        patch_size,
        patch_stride,
        train_gpt_mlp,
        d_ffn,
        dropout,
        embed,
        freq,
        n_classes: int = None,
    ):
        super().__init__()
        self.task_name = task_name

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_ffn = d_ffn
        self.n_patches = (n_steps + n_pred_steps - patch_size) // patch_stride + 2

        d_model = 768  # GPT2's hidden size

        self.padding_patch_layer = nn.ReplicationPad1d((0, patch_stride))
        self.enc_embedding = DataEmbedding(
            n_features,
            d_model,
            embed,
            freq,
            dropout,
        )

        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2",
            output_attentions=True,
            output_hidden_states=True,
        )
        self.gpt2.h = self.gpt2.h[:n_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if "ln" in name or "wpe" in name:  # or 'mlp' in name:
                param.requires_grad = True
            elif "mlp" in name and train_gpt_mlp:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if task_name == "long_term_forecast" or task_name == "short_term_forecast":
            self.predict_linear_pre = nn.Linear(n_steps, n_pred_steps + n_steps)
            self.predict_linear = nn.Linear(patch_size, n_features)
            self.ln = nn.LayerNorm(d_ffn)
            self.out_layer = nn.Linear(d_ffn, n_pred_features)
        elif self.task_name == "imputation":
            self.ln_proj = nn.LayerNorm(d_model)
            self.out_layer = nn.Linear(d_model, n_pred_features, bias=True)
        elif self.task_name == "anomaly_detection":
            self.ln_proj = nn.LayerNorm(d_ffn)
            self.out_layer = nn.Linear(d_ffn, n_pred_features, bias=True)
        elif self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(0.1)
            self.ln_proj = nn.LayerNorm(d_model * self.n_patches)
            self.out_layer = nn.Linear(d_model * self.n_patches, n_classes)
        else:
            raise ValueError("Invalid task name.")

    def imputation(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        mask: torch.Tensor,
    ):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state

        outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.n_pred_steps + self.n_steps, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.n_pred_steps + self.n_steps, 1))
        return dec_out

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
    ):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        enc_out = torch.nn.functional.pad(enc_out, (0, 768 - enc_out.shape[-1]))

        # enc_out = rearrange(enc_out, 'b l m -> b m l')
        # enc_out = self.padding_patch_layer(enc_out)
        # enc_out = enc_out.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        # enc_out = self.predict_linear(enc_out)
        # enc_out = rearrange(enc_out, 'b m n p -> b n (m p)')

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, : self.d_ffn]
        # dec_out = dec_out.reshape(B, -1)

        # dec_out = self.ln(dec_out)
        dec_out = self.out_layer(dec_out)
        # print(dec_out.shape)
        # dec_out = dec_out.reshape(B, self.pred_len + self.seq_len, -1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.n_pred_steps + self.n_steps, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.n_pred_steps + self.n_steps, 1))

        return dec_out

    def anomaly_detection(
        self,
        x_enc: torch.Tensor,
    ):
        # Normalization from Non-stationary Transformer
        seg_num = 25
        x_enc = rearrange(x_enc, "b (n s) m -> b n s m", s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, "b n s m -> b (n s) m")

        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        enc_out = torch.nn.functional.pad(x_enc, (0, 768 - x_enc.shape[-1]))

        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state

        outputs = outputs[:, :, : self.d_ffn]
        # outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer
        dec_out = rearrange(dec_out, "b (n s) m -> b n s m", s=seg_num)
        dec_out = dec_out * (stdev[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1))
        dec_out = dec_out + (means[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, "b n s m -> b (n s) m")

        return dec_out

    def classification(
        self,
        x_enc: torch.Tensor,
    ):
        B, L, M = x_enc.shape
        input_x = rearrange(x_enc, "b l m -> b m l")
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        input_x = rearrange(input_x, "b m n p -> b n (p m)")

        outputs = self.enc_embedding(input_x, None)

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        # outputs = self.dropout(outputs)
        outputs = self.out_layer(outputs)

        return outputs

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.n_pred_steps :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
