"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
from einops import rearrange

from .layers import TSLANet_layer, PatchEmbed, trunc_normal_


def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is removed
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]  # ids_keep: [bs x len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # x_kept: [bs x len_keep x dim]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)  # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
    )  # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is removed
    mask = torch.ones([bs, L], device=x.device)  # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


class BackboneTSLANet(nn.Module):
    def __init__(
        self,
        task_name: str,
        seq_len: int,
        num_channels: int,
        pred_len: int,
        n_layers: int,
        patch_size: int,
        emb_dim: int,
        dropout: float,
        mask_ratio: float,
        num_classes: int = None,
    ):
        super().__init__()

        self.task_name = task_name
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.stride = self.patch_size // 2
        num_patches = int((seq_len - self.patch_size) / self.stride + 1)

        # Layers/Networks
        dpr = [x.item() for x in torch.linspace(0, dropout, n_layers)]  # stochastic depth decay rule
        self.tsla_blocks = nn.ModuleList(
            [TSLANet_layer(dim=emb_dim, drop=dropout, drop_path=dpr[i]) for i in range(n_layers)]
        )

        if task_name == "classification":
            self.patch_embed = PatchEmbed(
                seq_len=seq_len, patch_size=patch_size, in_chans=num_channels, embed_dim=emb_dim
            )
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim), requires_grad=True)
            self.pos_drop = nn.Dropout(p=dropout)

            # Classifier head
            self.head = nn.Linear(emb_dim, num_classes)

            trunc_normal_(self.pos_embed, std=0.02)
            self.apply(self._init_weights)

        elif task_name in ["forecasting", "imputation", "anomaly_detection"]:
            self.input_layer = nn.Linear(self.patch_size, emb_dim)
            self.out_layer = nn.Linear(emb_dim * num_patches, pred_len)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in):
        if self.task_name == "classification":
            x = self.patch_embed(x_in)
            x = x + self.pos_embed
            x_patched = self.pos_drop(x)
        elif self.task_name in ["forecasting", "imputation", "anomaly_detection"]:
            x = rearrange(x_in, "b l m -> b m l")
            x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            x_patched = rearrange(x_patched, "b m n p -> (b m) n p")

        xb_mask, _, mask, _ = random_masking_3D(x_patched, mask_ratio=self.mask_ratio)
        mask = mask.bool()  # mask: [bs x num_patch x n_vars]

        if self.task_name in ["forecasting", "imputation", "anomaly_detection"]:
            xb_mask = self.input_layer(xb_mask)

        for tsla_blk in self.tsla_blocks:
            xb_mask = tsla_blk(xb_mask)

        if self.task_name == "classification":
            return xb_mask, x_patched
        elif self.task_name in ["forecasting", "imputation", "anomaly_detection"]:
            return xb_mask, self.input_layer(x_patched), mask

    def forward(self, x):
        if self.task_name == "classification":
            x = self.patch_embed(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)
        elif self.task_name in ["forecasting", "imputation", "anomaly_detection"]:
            B, L, M = x.shape
            x = rearrange(x, "b l m -> b m l")
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            x = rearrange(x, "b m n p -> (b m) n p")
            x = self.input_layer(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        if self.task_name == "classification":
            x = x.mean(1)
            outputs = self.head(x)
        elif self.task_name in ["forecasting", "imputation", "anomaly_detection"]:
            outputs = self.out_layer(x.reshape(B * M, -1))
            outputs = rearrange(outputs, "(b m) l -> b l m", b=B)

        return outputs
