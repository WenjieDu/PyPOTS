"""
The implementation of USAD (UnSupervised Anomaly Detection) for PyPOTS.

Paper: Audibert et al. (2020). USAD: UnSupervised Anomaly Detection on multivariate time series.
       KDD 2020. https://dl.acm.org/doi/10.1145/3394486.3403392

"""

# Created by omimajleta
# License: BSD-3-Clause

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pypots.anomaly_detection.base import BaseNNDetector
from pypots.data.dataset import BaseDataset
from pypots.nn.modules.loss import Criterion, MAE, MSE
from pypots.optim.adam import Adam
from pypots.optim.base import Optimizer


class _USADEncoder(nn.Module):
    """Shared encoder used by both AE1 and AE2 in USAD.

    Parameters
    ----------
    input_dim : int
        Flattened input dimension (n_steps * n_features).
    d_model : int
        Dimensionality of the encoder output layer.
    dropout : float
        Dropout rate.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _USADDecoder(nn.Module):
    """Individual decoder for AE1 or AE2 in USAD.

    Parameters
    ----------
    input_dim : int
        Flattened output dimension (n_steps * n_features).
    d_model : int
        Dimensionality of the encoder output layer.
    dropout : float
        Dropout rate.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model // 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class _USADNetwork(nn.Module):
    """The USAD network with a shared encoder and two separate decoders.

    As described in the original paper, AE1 and AE2 share the same encoder
    but have independent decoders.

    Parameters
    ----------
    n_steps : int
        Number of time steps in the input sequence.
    n_features : int
        Number of features in the input sequence.
    d_model : int
        Dimensionality of the encoder hidden layer.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.input_dim = n_steps * n_features

        # Shared encoder between AE1 and AE2
        self.encoder = _USADEncoder(self.input_dim, d_model, dropout)

        # Separate decoders for AE1 and AE2
        self.decoder1 = _USADDecoder(self.input_dim, d_model, dropout)
        self.decoder2 = _USADDecoder(self.input_dim, d_model, dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input through the shared encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, n_steps, n_features].

        Returns
        -------
        torch.Tensor
            Latent representation of shape [batch_size, d_model // 2].
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.encoder(x_flat)

    def decode1(self, z: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Decode latent representation using Decoder 1 (AE1)."""
        return self.decoder1(z).view(batch_size, self.n_steps, self.n_features)

    def decode2(self, z: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Decode latent representation using Decoder 2 (AE2)."""
        return self.decoder2(z).view(batch_size, self.n_steps, self.n_features)

    def forward(self, x: torch.Tensor):
        """Forward pass returning both AE1 and AE2 reconstructions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, n_steps, n_features].

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            x_hat1: reconstruction from AE1 (shared encoder + decoder1)
            x_hat2: reconstruction from AE2 (shared encoder + decoder2)
        """
        batch_size = x.shape[0]
        z = self.encode(x)
        x_hat1 = self.decode1(z, batch_size)
        x_hat2 = self.decode2(z, batch_size)
        return x_hat1, x_hat2


class USAD(BaseNNDetector):
    """USAD (UnSupervised Anomaly Detection) model for time series anomaly detection.

    USAD uses two autoencoders (AE1 and AE2). AE1 reconstructs the input,
    while AE2 reconstructs the output of AE1, making anomalies produce
    higher reconstruction errors.

    Parameters
    ----------
    n_steps : int
        Number of time steps in the input sequence.
    n_features : int
        Number of features in the input sequence.
    anomaly_rate : float
        Expected proportion of anomalies in the data (between 0 and 1).
    d_model : int
        Dimensionality of the encoder hidden layer. Default is 64.
    dropout : float
        Dropout rate. Default is 0.1.
    batch_size : int
        Batch size for training. Default is 32.
    epochs : int
        Number of training epochs. Default is 100.
    training_loss : Criterion or type
        Loss function for training. Default is MAE.
    validation_metric : Criterion or type
        Metric for validation. Default is MSE.
    optimizer : Optimizer or type
        Optimizer for training. Default is Adam.
    device : str or torch.device or list, optional
        Device to use for training. Default is None (auto-select).
    saving_path : str, optional
        Path to save the model. Default is None.
    verbose : bool
        Whether to print training progress. Default is True.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        anomaly_rate: float,
        d_model: int = 64,
        dropout: float = 0.1,
        batch_size: int = 32,
        epochs: int = 100,
        training_loss: Union[Criterion, type] = MAE,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
    ):
        super().__init__(
            anomaly_rate=anomaly_rate,
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            saving_path=saving_path,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        self.d_model = d_model
        self.dropout = dropout

        # Shared encoder + two separate decoders as per the original USAD paper
        self.model = _USADNetwork(
            n_steps=self.n_steps,
            n_features=self.n_features,
            d_model=self.d_model,
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

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the USAD model on the given dataset.

        Parameters
        ----------
        train_set : dict or str
            Training data. Must contain key "X" with shape
            [n_samples, n_steps, n_features].
        val_set : dict or str, optional
            Validation data. Default is None.
        file_type : str
            File type for loading data. Default is "hdf5".
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
            num_workers=self.num_workers,
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for raw_batch in train_loader:
                inputs = self._assemble_input_for_training(raw_batch)
                x = inputs["X"]

                x_hat1, x_hat2 = self.model(x)

                loss1 = self.training_loss(x, x_hat1)
                loss2 = self.training_loss(x, x_hat2)
                loss = loss1 + loss2

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
            Test data. Must contain key "X" with shape
            [n_samples, n_steps, n_features].
        file_type : str
            File type for loading data. Default is "hdf5".

        Returns
        -------
        dict
            Dictionary containing:
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
            num_workers=self.num_workers,
        )

        self.model.eval()
        all_scores = []

        with torch.no_grad():
            for raw_batch in test_loader:
                inputs = self._assemble_input_for_testing(raw_batch)
                x = inputs["X"]
                _, x_hat2 = self.model(x)
                scores = torch.mean((x - x_hat2) ** 2, dim=(1, 2))
                all_scores.append(scores.cpu().numpy())

        scores = np.concatenate(all_scores, axis=0)
        threshold = np.percentile(scores, (1 - self.anomaly_rate) * 100)
        anomaly_labels = (scores > threshold).astype(int)

        return {
            "anomaly_scores": scores,
            "anomaly_labels": anomaly_labels,
        }
