"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import torch
import torch.nn as nn
from transformers import (
    LlamaModel,
    LlamaTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    BertModel,
    BertTokenizer,
)

from .layers import ReprogrammingLayer
from ..patchtst.layers import PatchEmbedding, FlattenHead
from ..revin import RevIN

SUPPORTED_LLM = [
    "LLaMA",
    "GPT2",
    "BERT",
]

SUPPORTED_TASKS = [
    "long_term_forecast",
    "short_term_forecast",
    "imputation",
    "classification",
    "clustering",
]


class BackboneTimeLLM(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_pred_steps,
        n_layers,
        patch_size,
        patch_stride,
        d_model,
        d_ffn,
        d_llm,
        n_heads,
        llm_model_type,
        dropout,
        domain_prompt_content: str,
        task_name: str,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_steps = n_steps
        self.d_ffn = d_ffn
        self.d_llm = d_llm
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.task_name = task_name
        self.top_k = 5  # fixed value, the same as the official implementation

        assert n_steps > patch_size, "The length of the time series must be greater than the patch length."
        assert llm_model_type in SUPPORTED_LLM, f"The LLM model type must be one of {SUPPORTED_LLM}."
        assert task_name in SUPPORTED_TASKS, f"The task name must be one of {SUPPORTED_TASKS}."

        if llm_model_type == "LLaMA":
            self.llm_model = LlamaModel.from_pretrained(
                "huggyllama/llama-7b",
                num_hidden_layers=n_layers,
                output_attentions=True,
                output_hidden_states=True,
                # load_in_4bit=True
            )
            self.tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")

        elif llm_model_type == "GPT2":
            self.llm_model = GPT2Model.from_pretrained(
                "openai-community/gpt2",
                num_hidden_layers=n_layers,
                output_attentions=True,
                output_hidden_states=True,
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        elif llm_model_type == "BERT":
            self.llm_model = BertModel.from_pretrained(
                "google-bert/bert-base-uncased",
                num_hidden_layers=n_layers,
                output_attentions=True,
                output_hidden_states=True,
            )
            self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        else:
            raise Exception("LLM model is not defined")

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token

        # freeze the LLM model
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.patch_embedding = PatchEmbedding(
            d_model,
            patch_size,
            patch_stride,
            patch_stride,
            dropout,
            False,
        )

        self.domain_prompt_content = domain_prompt_content
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.n_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.n_tokens)
        self.reprogramming_layer = ReprogrammingLayer(d_model, n_heads, self.d_ffn, self.d_llm)
        self.n_patches = int((n_steps - self.patch_size) / self.patch_stride + 2)
        self.revin_layer = RevIN(n_features, affine=False)

        if self.task_name in ["long_term_forecast", "short_term_forecast", "imputation"]:
            self.output_projection = FlattenHead(
                d_ffn * self.n_patches,
                n_pred_steps,
                n_features,
                head_dropout=dropout,
            )
        else:
            raise NotImplementedError

    def calc_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def forward(self, x_enc, missing_mask=None):
        x_enc = self.revin_layer(x_enc, mode="norm")
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        if missing_mask is not None:
            missing_mask = missing_mask.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calc_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())

            if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.domain_prompt_content}"
                    "Task description: "
                    f"forecast the next {str(self.n_pred_steps)} steps given "
                    f"the previous {str(self.n_steps)} steps information; "
                    "Input statistics: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                    f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
                )
            elif self.task_name == "imputation":
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.domain_prompt_content}"
                    "Task description: "
                    f"given the observed information, "
                    f"impute the missing values that indicated as 0 in f{missing_mask[b].flatten()}; "
                    "Input statistics: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                    f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
                )
            else:
                raise NotImplementedError

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (bz, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()

        if os.getenv("ENABLE_AMP", False):
            enc_out = self.patch_embedding(x_enc.to(torch.bfloat16))
        else:
            enc_out = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, : self.d_ffn]

        dec_out = torch.reshape(dec_out, (-1, self.n_features, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        if self.task_name in ["long_term_forecast", "short_term_forecast", "imputation"]:
            dec_out = self.output_projection(dec_out[:, :, :, -self.n_patches :])
        else:
            raise NotImplementedError

        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.revin_layer(dec_out, mode="denorm")

        return dec_out
