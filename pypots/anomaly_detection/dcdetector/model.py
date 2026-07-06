"""
The implementation of DCdetector for PyPOTS.

Paper: Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun.
       DCdetector: Dual Attention Contrastive Representation Learning for
       Time Series Anomaly Detection.
       KDD 2023. https://dl.acm.org/doi/10.1145/3580305.3599295

"""

# Created by omimajleta
# License: BSD-3-Clause

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pypots.anomaly_detection.base import BaseNNDetector
from pypots.data.dataset import BaseDataset
from pypots.nn.modules.loss import Criterion, MAE, MSE
from pypots.optim.adam import Adam
from pypots.optim.base import Optimizer


class _PatchAttention(nn.Module):
    """Patched multi-head self-attention for one branch of DCdetector.

    Splits the time series into non-overlapping patches and applies
    multi-head attention within and across patches.

    Parameters
    ----------
    d_model : int
        Input and output feature dimensionality.
    n_heads : int
        Number of attention heads.
    patch_size : int
        Size of each time-series patch.
    dropout : float
        Dropout rate applied after attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        patch_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply patched attention to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch_size, n_steps, d_model].

        Returns
        -------
        torch.Tensor
            Output of shape [batch_size, n_steps, d_model].
        """
        batch_size, n_steps, d_model = x.shape

        # Pad sequence to be divisible by patch_size
        pad = (self.patch_size - n_steps % self.patch_size) % self.patch_size
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        n_padded = x.shape[1]
        n_patches = n_padded // self.patch_size

        # Reshape into patches: [B, n_patches, patch_size, d_model]
        x_patched = x.view(batch_size, n_patches, self.patch_size, d_model)
        # Flatten patches: [B * n_patches, patch_size, d_model]
        x_flat = x_patched.reshape(batch_size * n_patches, self.patch_size, d_model)

        # Multi-head attention within each patch
        q = self.q_proj(x_flat).view(
            batch_size * n_patches, self.patch_size, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x_flat).view(
            batch_size * n_patches, self.patch_size, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x_flat).view(
            batch_size * n_patches, self.patch_size, self.n_heads, self.head_dim
        ).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        # Reshape back: [B, n_steps_padded, d_model]
        out = out.transpose(1, 2).reshape(
            batch_size * n_patches, self.patch_size, d_model
        )
        out = self.out_proj(out)
        out = out.reshape(batch_size, n_padded, d_model)

        # Residual + LayerNorm
        out = self.norm(out + x)

        # Remove padding
        return out[:, :n_steps, :]


class _DCdetectorBlock(nn.Module):
    """A single DCdetector encoder block with dual-branch patched attention.

    The two branches use different patch sizes to capture multi-scale
    temporal patterns. Their representations are compared contrastively
    during training.

    Parameters
    ----------
    d_model : int
        Feature dimensionality.
    n_heads : int
        Number of attention heads.
    patch_size_1 : int
        Patch size for the first attention branch.
    patch_size_2 : int
        Patch size for the second attention branch.
    d_ff : int
        Inner dimensionality of the feed-forward layer.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        patch_size_1: int,
        patch_size_2: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.branch1 = _PatchAttention(d_model, n_heads, patch_size_1, dropout)
        self.branch2 = _PatchAttention(d_model, n_heads, patch_size_2, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """Forward pass through the dual-branch block.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch_size, n_steps, d_model].

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            out: merged output [batch_size, n_steps, d_model]
            rep1: branch-1 representation [batch_size, n_steps, d_model]
            rep2: branch-2 representation [batch_size, n_steps, d_model]
        """
        rep1 = self.branch1(x)
        rep2 = self.branch2(x)
        out = self.norm(self.ff(rep1 + rep2) + rep1 + rep2)
        return out, rep1, rep2


class _DCdetectorNetwork(nn.Module):
    """The core DCdetector network.

    Projects the input, applies stacked dual-branch encoder blocks, and
    accumulates per-timestep contrastive discrepancy scores.

    Parameters
    ----------
    n_steps : int
        Number of time steps in each input window.
    n_features : int
        Number of input features.
    d_model : int
        Internal feature dimensionality.
    n_heads : int
        Number of attention heads in each branch.
    n_layers : int
        Number of stacked encoder blocks.
    patch_size_1 : int
        Patch size for branch 1 (fine-grained scale).
    patch_size_2 : int
        Patch size for branch 2 (coarse scale).
    d_ff : int
        Feed-forward inner dimensionality.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        patch_size_1: int,
        patch_size_2: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features

        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([
            _DCdetectorBlock(d_model, n_heads, patch_size_1, patch_size_2, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """Forward pass computing dual representations and discrepancy.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch_size, n_steps, n_features].

        Returns
        -------
        tuple of (list, list, torch.Tensor)
            reps1: list of branch-1 representations per layer
            reps2: list of branch-2 representations per layer
            out: final encoder output [batch_size, n_steps, d_model]
        """
        h = self.input_proj(x)
        reps1, reps2 = [], []
        for block in self.blocks:
            h, r1, r2 = block(h)
            reps1.append(r1)
            reps2.append(r2)
        out = self.norm(h)
        return reps1, reps2, out


class DCdetector(BaseNNDetector):
    """DCdetector: Dual Attention Contrastive Representation Learning for
    Time Series Anomaly Detection.

    DCdetector uses a dual-branch patched attention structure to learn
    permutation-invariant representations. Anomalies are detected by
    measuring the discrepancy between the two branches' representations:
    normal points produce consistent representations across branches,
    while anomalous points produce divergent representations.

    Parameters
    ----------
    n_steps : int
        Number of time steps in each input sequence.
    n_features : int
        Number of features in the input sequence.
    anomaly_rate : float
        Expected proportion of anomalies in the data (between 0 and 1).
    d_model : int
        Internal feature dimensionality. Default is 64.
    n_heads : int
        Number of attention heads. Default is 4.
    n_layers : int
        Number of stacked encoder layers. Default is 2.
    patch_size_1 : int
        Patch size for the fine-grained attention branch. Default is 4.
    patch_size_2 : int
        Patch size for the coarse attention branch. Default is 8.
    d_ff : int
        Feed-forward inner dimensionality. Default is 128.
    dropout : float
        Dropout rate. Default is 0.1.
    batch_size : int
        Training batch size. Default is 32.
    epochs : int
        Number of training epochs. Default is 100.
    training_loss : Criterion or type
        Loss function used during training. Default is MAE.
    validation_metric : Criterion or type
        Metric used for validation. Default is MSE.
    optimizer : Optimizer or type
        Optimizer for parameter updates. Default is Adam.
    device : str or torch.device or list, optional
        Device for computation. Default is None (auto-select).
    saving_path : str, optional
        Path to save the trained model. Default is None.
    verbose : bool
        Whether to print training progress. Default is True.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        anomaly_rate: float,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        patch_size_1: int = 4,
        patch_size_2: int = 8,
        d_ff: int = 128,
        dropout: float = 0.1,
        batch_size: int = 32,
        epochs: int = 100,
        training_loss: Union[Criterion, type] = MAE,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        verbose: bool = True,
    ):
        super().__init__(
            anomaly_rate=anomaly_rate,
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            saving_path=saving_path,
            verbose=verbose,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.patch_size_1 = patch_size_1
        self.patch_size_2 = patch_size_2
        self.d_ff = d_ff
        self.dropout = dropout

        self.model = _DCdetectorNetwork(
            n_steps=self.n_steps,
            n_features=self.n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            patch_size_1=self.patch_size_1,
            patch_size_2=self.patch_size_2,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()
        assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data) -> dict:
        # BaseDataset returns a list: [indices, X, missing_mask]
        return {"X": data[1].to(self.device).float()}

    def _assemble_input_for_validating(self, data) -> dict:
        return {"X": data[1].to(self.device).float()}

    def _assemble_input_for_testing(self, data) -> dict:
        return {"X": data[1].to(self.device).float()}

    @staticmethod
    def _contrastive_loss(reps1: list, reps2: list) -> torch.Tensor:
        """Compute the contrastive discrepancy loss between dual branches.

        The loss is the mean cosine similarity between the two branches'
        representations across all layers, encouraging them to diverge
        for anomalous inputs and converge for normal inputs.

        Parameters
        ----------
        reps1 : list of torch.Tensor
            Branch-1 representations, one per encoder layer.
        reps2 : list of torch.Tensor
            Branch-2 representations, one per encoder layer.

        Returns
        -------
        torch.Tensor
            Scalar contrastive loss value.
        """
        loss = torch.tensor(0.0, device=reps1[0].device)
        for r1, r2 in zip(reps1, reps2):
            r1_norm = F.normalize(r1, dim=-1)
            r2_norm = F.normalize(r2, dim=-1)
            # Maximize discrepancy → minimize negative cosine similarity
            loss = loss + (r1_norm * r2_norm).sum(dim=-1).mean()
        return loss / len(reps1)

    @staticmethod
    def _anomaly_score(reps1: list, reps2: list) -> torch.Tensor:
        """Compute per-sample anomaly scores from dual-branch discrepancy.

        Parameters
        ----------
        reps1 : list of torch.Tensor
            Branch-1 representations per layer [B, n_steps, d_model].
        reps2 : list of torch.Tensor
            Branch-2 representations per layer [B, n_steps, d_model].

        Returns
        -------
        torch.Tensor
            Anomaly scores of shape [batch_size].
        """
        score = torch.zeros(reps1[0].shape[0], device=reps1[0].device)
        for r1, r2 in zip(reps1, reps2):
            diff = (r1 - r2) ** 2
            score = score + diff.mean(dim=(1, 2))
        return score / len(reps1)

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the DCdetector model on the given dataset.

        Parameters
        ----------
        train_set : dict or str
            Training data containing key "X" of shape
            [n_samples, n_steps, n_features].
        val_set : dict or str, optional
            Validation data. Default is None.
        file_type : str
            File format for loading from disk. Default is "hdf5".
        """
        if not isinstance(train_set, dict):
            raise TypeError("train_set must be a dictionary")
        if "X" not in train_set:
            raise KeyError("train_set must contain key 'X'")

        train_dataset = BaseDataset(
            data=train_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for raw_batch in train_loader:
                inputs = self._assemble_input_for_training(raw_batch)
                x = inputs["X"]

                reps1, reps2, _ = self.model(x)
                loss = self._contrastive_loss(reps1, reps2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            if self.verbose:
                print(f"Epoch {epoch:03d} - Training Loss: {avg_loss:.4f}")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        **kwargs,
    ) -> dict:
        """Detect anomalies in the given test dataset.

        Parameters
        ----------
        test_set : dict or str
            Test data containing key "X" of shape
            [n_samples, n_steps, n_features].
        file_type : str
            File format for loading from disk. Default is "hdf5".

        Returns
        -------
        dict
            - "anomaly_scores": float array of shape [n_samples]
            - "anomaly_labels": int array of shape [n_samples] (1 = anomaly)
        """
        test_dataset = BaseDataset(
            data=test_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()
        all_scores = []

        with torch.no_grad():
            for raw_batch in test_loader:
                inputs = self._assemble_input_for_testing(raw_batch)
                x = inputs["X"]
                reps1, reps2, _ = self.model(x)
                scores = self._anomaly_score(reps1, reps2)
                all_scores.append(scores.cpu().numpy())

        scores = np.concatenate(all_scores, axis=0)
        threshold = np.percentile(scores, (1 - self.anomaly_rate) * 100)
        anomaly_labels = (scores > threshold).astype(int)

        return {
            "anomaly_scores": scores,
            "anomaly_labels": anomaly_labels,
        }
