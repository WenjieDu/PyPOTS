"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResidualStack


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        embedding_dim,
        compression_factor,
    ):
        super().__init__()
        assert compression_factor in [4, 8, 12, 16], "compression_factor must be one of [4, 8, 12, 16]"

        self._conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv1d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._pre_vq_conv = nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
        )

        if compression_factor == 4:
            self._conv_3 = nn.Conv1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        elif compression_factor == 8:
            self._conv_A = nn.Conv1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self._conv_3 = nn.Conv1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        elif compression_factor == 12:
            self._conv_3 = nn.Conv1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=3,
                padding=1,
            )
            self._conv_4 = nn.Conv1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        elif compression_factor == 16:
            self._conv_A = nn.Conv1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self._conv_B = nn.Conv1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self._conv_3 = nn.Conv1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, inputs, compression_factor):
        x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

        x = self._conv_1(x)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        if compression_factor == 4:
            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 8:
            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 12:
            x = self._conv_3(x)
            x = F.relu(x)

            x = self._conv_4(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 16:
            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_B(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        compression_factor,
    ):
        super().__init__()
        assert compression_factor in [4, 8, 12, 16], "compression_factor must be one of [4, 8, 12, 16]"

        self._conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        if compression_factor == 4:
            self._conv_trans_1 = nn.ConvTranspose1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self._conv_trans_2 = nn.ConvTranspose1d(
                in_channels=num_hiddens // 2,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            )

        elif compression_factor == 8:
            self._conv_trans_A = nn.ConvTranspose1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self._conv_trans_1 = nn.ConvTranspose1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self._conv_trans_2 = nn.ConvTranspose1d(
                in_channels=num_hiddens // 2,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            )

        elif compression_factor == 12:
            # To get the correct shape back the kernel size has to be 5 not 4
            self._conv_trans_2 = nn.ConvTranspose1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=5,
                stride=3,
                padding=1,
            )

            self._conv_trans_3 = nn.ConvTranspose1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )

            self._conv_trans_4 = nn.ConvTranspose1d(
                in_channels=num_hiddens // 2,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            )

        elif compression_factor == 16:
            self._conv_trans_A = nn.ConvTranspose1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=2,
                padding=1,
            )

            self._conv_trans_B = nn.ConvTranspose1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=2,
                padding=1,
            )

            self._conv_trans_1 = nn.ConvTranspose1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )

            self._conv_trans_2 = nn.ConvTranspose1d(
                in_channels=num_hiddens // 2,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            )

    def forward(self, inputs, compression_factor):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)

        if compression_factor == 4:
            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)

        elif compression_factor == 8:
            x = self._conv_trans_A(x)
            x = F.relu(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)

        elif compression_factor == 12:
            x = self._conv_trans_2(x)
            x = F.relu(x)

            x = self._conv_trans_3(x)
            x = F.relu(x)

            x = self._conv_trans_4(x)

            return torch.squeeze(x)

        elif compression_factor == 16:
            x = self._conv_trans_A(x)
            x = F.relu(x)

            x = self._conv_trans_B(x)
            x = F.relu(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
    ):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (
            loss,
            quantized.permute(0, 2, 1).contiguous(),
            perplexity,
            self._embedding.weight,
            encoding_indices,
            encodings,
        )


class VQVAE(nn.Module):
    def __init__(
        self,
        block_hidden_size,
        num_residual_layers,
        res_hidden_size,
        embedding_dim,
        num_embeddings,
        commitment_cost,
        compression_factor,
    ):
        super().__init__()

        self.vq = VectorQuantizer(
            num_embeddings,
            embedding_dim,
            commitment_cost,
        )
        self.encoder = Encoder(
            1,
            block_hidden_size,
            num_residual_layers,
            res_hidden_size,
            embedding_dim,
            compression_factor,
        )
        self.decoder = Decoder(
            embedding_dim,
            block_hidden_size,
            num_residual_layers,
            res_hidden_size,
            compression_factor,
        )
        self.compression_factor = compression_factor

    def forward(self, X):
        z = self.encoder(X, self.compression_factor)
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
        data_recon = self.decoder(quantized, self.compression_factor)
        return data_recon, vq_loss, perplexity, embedding_weight, encoding_indices, encodings
