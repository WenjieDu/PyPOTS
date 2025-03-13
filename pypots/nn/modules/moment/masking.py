"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Optional

import torch


class Masking:
    def __init__(
        self,
        mask_ratio: float = 0.3,
        patch_size: int = 8,
        stride: Optional[int] = None,
    ):
        """
        Indices with 0 mask are hidden, and with 1 are observed.
        """
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.patch_stride = patch_size if stride is None else stride

    @staticmethod
    def convert_seq_to_patch_view(
        mask: torch.Tensor,
        patch_size: int = 8,
        stride: Optional[int] = None,
    ):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x seq_len]
        Output
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        stride = patch_size if stride is None else stride
        mask = mask.unfold(dimension=-1, size=patch_size, step=stride)
        # mask : [batch_size x n_patches x patch_size]
        return (mask.sum(dim=-1) == patch_size).long()

    @staticmethod
    def convert_patch_to_seq_view(
        mask: torch.Tensor,
        patch_size: int = 8,
    ):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        return mask.repeat_interleave(patch_size, dim=-1)

    def generate_mask(
        self,
        x: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x n_patches x patch_size] or
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len] or
            [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        if x.ndim == 4:
            return self._mask_patch_view(x, input_mask=input_mask)
        elif x.ndim == 3:
            return self._mask_seq_view(x, input_mask=input_mask)

    def _mask_patch_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x n_patches x patch_size]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        input_mask = self.convert_seq_to_patch_view(input_mask, self.patch_size, self.patch_stride)
        n_observed_patches = input_mask.sum(dim=-1, keepdim=True)  # batch_size x 1

        batch_size, _, n_patches, _ = x.shape
        len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()
        noise = torch.rand(
            batch_size, n_patches, device=x.device
        )  # noise in [0, 1], batch_size x n_channels x n_patches
        noise = torch.where(input_mask == 1, noise, torch.ones_like(noise))  # only keep the noise of observed patches

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [batch_size x n_patches]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([batch_size, n_patches], device=x.device)  # mask: [batch_size x n_patches]
        for i in range(batch_size):
            mask[i, : len_keep[i]] = 1

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.long()

    def _mask_seq_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        mask = self._mask_patch_view(x, input_mask=input_mask)
        return self.convert_patch_to_seq_view(mask, self.patch_size).long()
