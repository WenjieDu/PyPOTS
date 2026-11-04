"""LIFTS for partially observed time-series forecasting."""

from __future__ import annotations

from torch.utils.data import DataLoader

from ...data.checking import key_in_data_set
from ...data.dataset import BaseDataset
from ...nn.modules.lifts.initialization import LIFTSInitializationMixin
from ...nn.modules.lifts.training import ReferenceTrainingMixin
from ..base import BaseNNForecaster
from .core import _LIFTS


class LIFTS(ReferenceTrainingMixin, LIFTSInitializationMixin, BaseNNForecaster):
    """Latent-driven Imputation and Forecasting in Time Series :cite:`wu2026lifts`.

    Parameters
    ----------
    n_steps, n_features : int
        Input sequence length and feature count. Inputs use
        [n_samples, n_steps, n_features] with NaN for missing entries.
    n_pred_steps : int
        Number of future steps. Forecasts have the same features as the input.
        X_pred is checked for API compatibility; the original training objective
        uses X only. Forecast prediction does not require X_pred.
    d_latent : int, optional
        Latent dimension. Defaults to n_features for n_features <= 50,
        otherwise max(50, n_features // 2), rounding the half down.
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
        Monte Carlo repetitions for the forecast training objective (default 1).
    lambda_enc : float
        Encoder calibration weight; this faithful default path supports 0 only.
    batch_size, epochs, patience : int
        Training batch size and epoch count. Original LIFTS runs all epochs;
        patience is retained for API compatibility and does not stop training.
    validation_metric : Criterion or type
        Retained for API compatibility; checkpoints follow original training loss.
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
    v1 smoother, v2 decoder, MLP emissions), using the original joint-training
    schedule, a single Adam optimizer, and training-loss checkpoint selection.
    Optional research variants and encoder calibration are not included. Standardize inputs using training-only statistics before
    fitting, as with other PyPOTS models. The estimator does not normalize data.
    """

    _lifts_core = _LIFTS

    def __init__(self, n_steps: int, n_features: int, n_pred_steps: int, **kwargs):
        self.n_pred_steps = n_pred_steps
        super().__init__(n_steps, n_features, **kwargs)

    def _make_dataset(self, data, file_type, validating=False, predicting=False):
        dataset = BaseDataset(
            data,
            return_X_ori=False,
            return_X_pred=not predicting,
            return_y=False,
            file_type=file_type,
        )
        if dataset.n_samples < 1 or (dataset.n_steps, dataset.n_features) != (self.n_steps, self.n_features):
            raise ValueError("X must be nonempty with shape [n_samples, n_steps, n_features].")
        if not predicting and (dataset.n_pred_steps, dataset.n_pred_features) != (
            self.n_pred_steps,
            self.n_features,
        ):
            raise ValueError("X_pred must have shape [n_samples, n_pred_steps, n_features].")
        return dataset

    def fit(self, train_set, val_set=None, file_type="hdf5"):
        """Run all epochs; select minimum training joint loss after the first quarter.

        val_set is checked for API compatibility but never selects the checkpoint.
        """
        train_dataset = self._make_dataset(train_set, file_type)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_pred", val_set):
                raise ValueError("val_set must contain 'X_pred' for validation.")
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

    def predict(self, test_set, file_type="hdf5", n_sampling_times=100):
        """Return point estimates and Monte Carlo samples.

        forecasting has shape [batch, n_pred_steps, n_features];
        forecasting_samples adds a sample axis after batch. Point estimates
        are means of these samples. By default, generate 100 samples.
        Data must be standardized using training-only statistics.
        """
        self._make_dataset(test_set, file_type, predicting=True)
        return super().predict(test_set, file_type, n_sampling_times=n_sampling_times)
