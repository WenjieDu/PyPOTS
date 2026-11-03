"""LIFTS for partially observed time-series imputation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ...data.checking import key_in_data_set
from ...data.dataset import BaseDataset
from ...nn.modules.loss import MSE, Criterion
from ...optim import Adam
from ...optim.base import Optimizer
from ..base import BaseNNImputer
from .core import _LIFTS


class LIFTS(BaseNNImputer):
    """Latent-driven Imputation and Forecasting in Time Series :cite:`wu2026lifts`.

    Parameters
    ----------
    n_steps, n_features : int
        Input sequence length and feature count. Inputs use
        [n_samples, n_steps, n_features] with NaN for missing entries.
    d_latent : int, optional
        Latent dimension; defaults to n_features.
    d_model, n_heads, d_ffn : int
        Hidden width, attention heads, and feed-forward width. Each head's
        width (d_model / n_heads) must be even for rotary embeddings.
    n_layers, n_inner_layers : int
        Number of filter blocks and shared-weight applications per block.
    self_mask_rate : float
        Probability of hiding an observed entry for the auxiliary objective.
    self_mask_weight, mask_weight : float
        Weights for self-masked imputation and observation-mask prediction.
    n_sampling_times_train : int
        Monte Carlo repetitions for the forecast training objective.
    batch_size, epochs, patience : int
        Training batch size, maximum epochs, and early-stopping patience.
    validation_metric : Criterion or type
        Metric for selecting checkpoints; defaults to MSE.
    optimizer : Optimizer or type
        PyPOTS optimizer; defaults to Adam with learning rate 0.001.
    num_workers : int
        Data-loader workers; zero loads in the main process.
    device : str, torch.device, list, optional
        Device(s) supported by PyPOTS; defaults to CUDA if available, else CPU.
    saving_path : str, optional
        Directory for checkpoints and TensorBoard logs.
    model_saving_strategy : str, optional
        One of None, 'best', 'better', or 'all'.
    verbose : bool
        Whether to print training logs.

    Notes
    -----
    Implements the reference default joint-training configuration (v1 filter,
    v1 smoother, v2 decoder, MLP emissions), using the standard NN training
    loop and one optimizer. Optional research variants and encoder calibration
    are not included. Standardize inputs using training-only statistics before
    fitting, as with other PyPOTS models. The estimator does not normalize data.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        d_latent: int | None = None,
        d_model: int = 128,
        n_heads: int = 4,
        d_ffn: int = 512,
        n_layers: int = 3,
        n_inner_layers: int = 2,
        self_mask_rate: float = 0.2,
        self_mask_weight: float = 5.0,
        mask_weight: float = 1.0,
        n_sampling_times_train: int = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int | None = None,
        validation_metric: Criterion | type = MSE,
        optimizer: Optimizer | type = Adam,
        num_workers: int = 0,
        device: str | torch.device | list | None = None,
        saving_path: str | None = None,
        model_saving_strategy: str | None = "best",
        verbose: bool = True,
    ):
        d_latent = n_features if d_latent is None else d_latent
        dimensions = {
            "n_steps": n_steps,
            "n_features": n_features,
            "d_latent": d_latent,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ffn": d_ffn,
            "n_layers": n_layers,
            "n_inner_layers": n_inner_layers,
            "n_sampling_times_train": n_sampling_times_train,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("d_model must be divisible by n_heads with an even head dimension.")
        if not 0 < self_mask_rate < 1:
            raise ValueError("self_mask_rate must be between 0 and 1, exclusively.")
        if not 0 < self_mask_weight < float("inf") or not 0 <= mask_weight < float("inf"):
            raise ValueError("self_mask_weight must be positive and mask_weight nonnegative, both finite.")
        super().__init__(
            training_loss=Criterion,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )
        for name, value in dimensions.items():
            setattr(self, name, value)
        self.self_mask_rate = self_mask_rate
        self.self_mask_weight = self_mask_weight
        self.mask_weight = mask_weight
        self.model = _LIFTS(
            n_steps=n_steps,
            n_features=n_features,
            d_latent=d_latent,
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            n_layers=n_layers,
            n_inner_layers=n_inner_layers,
            self_mask_rate=self_mask_rate,
            self_mask_weight=self_mask_weight,
            mask_weight=mask_weight,
            n_sampling_times_train=n_sampling_times_train,
            validation_metric=self.validation_metric,
        )
        self._print_model_size()
        self._send_model_to_given_device()
        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else optimizer()
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError("optimizer must be a PyPOTS Optimizer or Optimizer class.")
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data):
        _, X, mask = self._send_data_to_given_device(data)
        return {"X": X, "missing_mask": mask}

    def _assemble_input_for_validating(self, data):
        _, X, mask, X_ori, indicating_mask = self._send_data_to_given_device(data)
        return {"X": X, "missing_mask": mask, "X_ori": X_ori, "indicating_mask": indicating_mask}

    def _assemble_input_for_testing(self, data):
        return self._assemble_input_for_training(data)

    def _make_dataset(self, data, file_type, validating=False, predicting=False):
        dataset = BaseDataset(
            data,
            return_X_ori=validating,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        if dataset.n_samples < 1 or (dataset.n_steps, dataset.n_features) != (self.n_steps, self.n_features):
            raise ValueError("X must be nonempty with shape [n_samples, n_steps, n_features].")
        return dataset

    def fit(self, train_set, val_set=None, file_type="hdf5"):
        """Train on a dict or HDF5 path, optionally selecting checkpoints on val_set."""
        train_dataset = self._make_dataset(train_set, file_type)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for validation.")
            val_dataset = self._make_dataset(val_set, file_type, validating=True)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        self._train_model(train_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    def predict(self, test_set, file_type="hdf5", n_sampling_times=1):
        """Return point estimates and Monte Carlo samples.

        imputation has shape [batch, n_steps, n_features];
        imputation_samples adds a sample axis after batch. Point estimates
        are means of these samples. Increase n_sampling_times for uncertainty
        estimates. Data must be standardized using training-only statistics.
        """
        self._make_dataset(test_set, file_type, predicting=True)
        return super().predict(test_set, file_type, n_sampling_times=n_sampling_times)
