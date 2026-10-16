"""
The implementation of TS-ICL for the partially-observed time-series imputation task.

"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause

import warnings
from typing import Optional, Union

import torch

from ...nn.modules.tsicl import build_tsicl_network, fetch_X, load_tsicl_checkpoint, resolve_inference_device
from ..base import BaseImputer
from .core import impute_with_tsicl


class TSICL(BaseImputer):
    """The PyPOTS wrapper of the TS-ICL time-series foundation model :cite:`lenaour2026tsicl`.

    TS-ICL is a pretrained time-indexed foundation model that imputes **zero-shot**,
    without any gradient step: its architecture and pretrained checkpoint are vendored
    into PyPOTS (see ``pypots.nn.modules.tsicl``), and ``fit()`` is a no-op, the model
    having no fine-tuning procedure in the original implementation this is ported from.

    Warnings
    --------
    While the TS-ICL implementation in PyPOTS is licensed under the BSD-3-Clause license,
    the official pretrained TS-ICL weights hosted on Hugging Face (and downloaded by default,
    see ``allow_auto_download``) are licensed by EDF SA under the **TS-ICL Non-Commercial
    License**: they may only be used for non-commercial purposes. See
    https://huggingface.co/taharnbl/TS-ICL for the license terms, and contact
    tsicl-contact@edf.fr for commercial licensing.

    Parameters
    ----------
    model_path :
        Path to a local TS-ICL checkpoint. If None, the checkpoint is downloaded
        from the Hugging Face Hub (repo ``taharnbl/TS-ICL``) on first use and cached
        locally, the same way PyPOTS' other pretrained backbones (e.g. GPT4TS, MOMENT)
        fetch their weights.

    checkpoint_version :
        Checkpoint release to fetch from the Hub when ``model_path`` is None.

    allow_auto_download :
        Whether to automatically download the checkpoint from the Hub when it isn't
        already cached locally / at ``model_path``.

    batch_size :
        Number of univariate series imputed per forward pass. Note that a
        multivariate sample contributes ``n_features`` series (see the note on
        channel independence below).

    point_estimator :
        How the point estimate is derived from the predicted quantiles, either
        ``'median'`` or ``'mean'``. The median is optimal under absolute error,
        so it is the better default when evaluating with MAE.

    quantile_levels :
        Quantile levels requested from the model. Only used to derive the point
        estimate here; PyPOTS' imputation contract returns point values only.

    device :
        The device for the model to run on.

    saving_path :
        The path for automatically saving the model when calling ``fit()``.
        Will not save if not given.

    verbose :
        Whether to print out the logs during the process.

    Notes
    -----
    **Channel independence.** TS-ICL models one series at a time. A multivariate
    sample of shape ``(n_steps, n_features)`` is therefore imputed feature by
    feature, and cross-feature correlation is not exploited. On datasets where
    features are strongly correlated this is a genuine handicap against
    multivariate models such as SAITS or ImputeFormer, and results should be
    read with that in mind.

    **Sequence length.** ``n_steps`` must not exceed the model's maximum context
    length (4096 for ``tsicl-v1``); TS-ICL has no imputation rollout beyond it.

    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        checkpoint_version: str = "tsicl-v1.ckpt",
        allow_auto_download: bool = True,
        batch_size: int = 32,
        point_estimator: str = "median",
        quantile_levels: Optional[list[float]] = None,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        verbose: bool = True,
    ):
        super().__init__(device=device, saving_path=saving_path, verbose=verbose)

        assert point_estimator in ["median", "mean"]

        self.batch_size = batch_size
        self.point_estimator = point_estimator
        self.quantile_levels = quantile_levels or [0.1, 0.3, 0.5, 0.7, 0.9]

        checkpoint = load_tsicl_checkpoint(model_path, checkpoint_version, allow_auto_download)
        # zero-shot inference runs on a single device
        self.device = resolve_inference_device(self.device)
        self.model = build_tsicl_network(checkpoint, "imputer").to(self.device)
        self.max_context_length = checkpoint["config"]["max_context_len"]

    def _check_len(self, n_steps: int) -> None:
        if n_steps > self.max_context_length:
            raise ValueError(
                f"n_steps={n_steps} exceeds TS-ICL's maximum context length "
                f"({self.max_context_length}). Split the sequences before imputing."
            )

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Do nothing: TS-ICL is used zero-shot and has no fine-tuning procedure.

        Warnings
        --------
        TS-ICL has no parameter to train in this integration. Please run func
        ``predict()`` directly.

        """
        warnings.warn(
            "TS-ICL is a pretrained foundation model used zero-shot here and has no "
            "fine-tuning procedure in this integration. Please run func `predict()` directly.",
            stacklevel=2,
        )
        # keep PyPOTS' contract: the model is saved into `saving_path` (if given) once "trained"
        self._auto_save_model_if_necessary()

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        **kwargs,
    ) -> dict:
        X = fetch_X(test_set, file_type)
        self._check_len(X.shape[1])

        imputed_data = impute_with_tsicl(
            imputer=self.model,
            X=X,
            batch_size=self.batch_size,
            device=self.device,
            point_estimator=self.point_estimator,
            quantile_levels=self.quantile_levels,
        )
        return {"imputation": imputed_data}
