"""Shared setup for the two LIFTS task APIs; no training or sampling policy."""

# Created by Sanyou Wu <sw4180@cumc.columbia.edu>
# License: BSD-3-Clause

from __future__ import annotations

import torch

from ....optim import Adam
from ....optim.base import Optimizer
from ..loss import MSE, Criterion
from .training import reference_parameter_order


class _LIFTSDataParallel(torch.nn.DataParallel):
    def forward(self, inputs, **kwargs):
        # Scatter the batch and options together. Separate positional inputs
        # and keyword options can create an empty replica on a short batch.
        return super().forward(inputs=inputs, **kwargs)


def initialize_lifts(estimator, base_init, core_type, options):
    """Initialize either task without duplicating model/optimizer configuration.

    The public constructors retain explicit signatures. ``options`` is their
    local argument mapping; only the documented configuration keys are read.
    """
    if options["lambda_enc"] != 0:
        raise ValueError("This reference training path supports lambda_enc=0 only.")
    estimator.lambda_enc = float(options["lambda_enc"])
    features, latent = options["n_features"], options["d_latent"]
    if latent is None:
        latent = features if features <= 50 else max(50, features // 2)
    dimensions = {
        name: options[name]
        for name in (
            "n_steps",
            "n_features",
            "d_model",
            "n_heads",
            "d_ffn",
            "n_layers",
            "n_inner_layers",
            "n_sampling_times_train",
        )
    }
    dimensions["d_latent"] = latent
    if "n_pred_steps" in options:
        dimensions["n_pred_steps"] = options["n_pred_steps"]
    for name, value in dimensions.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    width, heads = dimensions["d_model"], dimensions["n_heads"]
    if width % heads or (width // heads) % 2:
        raise ValueError("d_model must be divisible by n_heads with an even head dimension.")
    if not 0 < options["self_mask_rate"] < 1:
        raise ValueError("self_mask_rate must be between 0 and 1, exclusively.")
    if not 0 < options["self_mask_weight"] < float("inf") or not 0 <= options["mask_weight"] < float("inf"):
        raise ValueError("self_mask_weight must be positive and mask_weight nonnegative, both finite.")
    base_init(
        training_loss=Criterion,
        **{
            name: options[name]
            for name in (
                "validation_metric",
                "batch_size",
                "epochs",
                "patience",
                "num_workers",
                "device",
                "saving_path",
                "model_saving_strategy",
                "verbose",
            )
        },
    )
    for name, value in dimensions.items():
        setattr(estimator, name, value)
    objective = {name: options[name] for name in ("self_mask_rate", "self_mask_weight", "mask_weight")}
    for name, value in objective.items():
        setattr(estimator, name, value)
    core_dimensions = dict(dimensions)
    if "n_pred_steps" in dimensions:
        core_dimensions["n_steps"] += dimensions["n_pred_steps"]
    estimator.model = core_type(**core_dimensions, **objective, validation_metric=estimator.validation_metric)
    estimator._print_model_size()
    estimator._send_model_to_given_device()
    optimizer = options["optimizer"]
    estimator.optimizer = optimizer if isinstance(optimizer, Optimizer) else optimizer()
    if not isinstance(estimator.optimizer, Optimizer):
        raise TypeError("optimizer must be a PyPOTS Optimizer or Optimizer class.")
    core = estimator.model
    if isinstance(core, torch.nn.DataParallel):
        core = core.module
    estimator.optimizer.init_optimizer(reference_parameter_order(core.backbone))


class LIFTSInitializationMixin:
    """Common public configuration for LIFTS task estimators."""

    def _send_model_to_given_device(self):
        super()._send_model_to_given_device()
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = _LIFTSDataParallel(self.model.module, device_ids=self.model.device_ids)

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
        lambda_enc: float = 0.0,
        batch_size: int = 64,
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
        options = locals()
        if hasattr(self, "n_pred_steps"):
            options = {**options, "n_pred_steps": self.n_pred_steps}
        initialize_lifts(self, super().__init__, self._lifts_core, options)
