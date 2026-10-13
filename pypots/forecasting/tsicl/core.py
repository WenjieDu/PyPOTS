"""
The core forecasting logic of TS-ICL, ported from the (zero-shot, gradient-free)
``TSICL.forecast()`` pipeline of the original implementation. Covariates are out of
scope here since PyPOTS' forecasting contract has no notion of them. The checkpoint
loading, batching, and context/target grid plumbing shared with
``pypots.imputation.tsicl`` live in ``pypots.nn.modules.tsicl``.

Notes
-----
Official pretrained weights hosted on Hugging Face are subject to the TS-ICL model license.

"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause

from typing import Optional

import numpy as np
import torch
from torch import nn

from ...nn.modules.tsicl import (
    CustomStandardScaler,
    complete_nans,
    flatten_channel_independent,
    make_grid,
    nan_row_fallback,
    prepare_context_tensors,
    resolve_quantile_selection,
    run_batched_point_estimate,
    unflatten_channel_independent,
)


def _rollout_forecast_batch(
    forecaster: nn.Module,
    grid: torch.Tensor,
    series_c: torch.Tensor,
    prediction_length: int,
    max_context_length: int,
    max_target_length: int,
    this_context_length: int,
) -> torch.Tensor:
    """Forecast ``prediction_length`` steps ahead, autoregressively rolling the context
    forward whenever ``prediction_length`` exceeds the model's per-pass horizon cap
    (``max_target_length``, e.g. 672 for ``tsicl-v1``). Ported (covariate-free) from
    ``TSICL._rollout_f`` + ``TSICL._get_context_target_coords_f`` + ``TSICL._predict_batch``.

    Coordinates are relative to the (possibly extended) context, not absolute: every rollout
    step queries the grid positions right after the context window, and the context is
    grown by the (raw, all-quantile) mean of that step's prediction before the next step.
    """
    grid_threshold = max_context_length
    quantile_chunks = []
    remaining = prediction_length

    while remaining > 0:
        pred_len = min(max_target_length, remaining)
        lookback_len = min(series_c.shape[1], this_context_length)

        coords_c = grid[:, grid_threshold - lookback_len : grid_threshold]
        coords_t = grid[:, grid_threshold : grid_threshold + pred_len]
        series_c_window = series_c[:, -lookback_len:]

        context = prepare_context_tensors(grid=coords_c, series_c=series_c_window)
        series_c_ctx, coords_c_ctx = context["series_c"], context["coords_c"]
        filled = complete_nans(series_c_ctx, coords_c_ctx)
        series_c_ctx, coords_c_ctx = filled["values"], filled["coords"]

        scaler = CustomStandardScaler(dim=1, epsilon=1e-5)
        scaler.fit(series_c_ctx)
        series_c_norm = scaler.transform(series_c_ctx)

        query_coords = torch.cat([coords_c_ctx, coords_t], dim=1)
        quantiles, _ = forecaster(
            series=series_c_norm,
            coords=coords_c_ctx,
            target_coords=query_coords,
            undo_asinh_transform=True,
        )
        quantiles = scaler.inv_transform(quantiles)
        quantile_chunks.append(quantiles)

        # extend the context with this step's forecast (raw mean across all quantile
        # levels, matching the original rollout, not just the requested point estimator)
        series_c = torch.cat([series_c, quantiles.mean(dim=-1, keepdim=True)], dim=1)
        remaining -= pred_len

    return torch.cat(quantile_chunks, dim=1)


def forecast_with_tsicl(
    forecaster: nn.Module,
    X: np.ndarray,
    prediction_length: int,
    max_context_length: int,
    max_target_length: int,
    batch_size: int,
    device: torch.device,
    point_estimator: str = "median",
    quantile_levels: Optional[list[float]] = None,
    context_length: Optional[int] = None,
) -> np.ndarray:
    """Forecast `prediction_length` steps beyond a `(n_samples, n_steps, n_features)` array
    (which may contain NaNs), feature by feature (TS-ICL models one series at a time; see the
    channel-independence note in ``TSICL``'s docstring). Unlike imputation, the lookback window
    is silently truncated to the most recent ``context_length`` steps rather than rejected if
    longer than the model supports, matching the original pipeline's forecasting behavior.
    """
    quantile_indices, median_idx, point_estimator = resolve_quantile_selection(
        forecaster, point_estimator, quantile_levels
    )
    this_context_length = min(max_context_length, context_length) if context_length else max_context_length
    grid_len = max_context_length + max_target_length

    flat, n_samples, n_features = flatten_channel_independent(X)
    forecaster = forecaster.to(device)

    def predict_batch_fn(series_c: torch.Tensor) -> torch.Tensor:
        grid = make_grid(grid_len, num_samples=series_c.shape[0]).to(device)
        return _rollout_forecast_batch(
            forecaster, grid, series_c, prediction_length, max_context_length, max_target_length, this_context_length
        )

    out = run_batched_point_estimate(
        flat, prediction_length, batch_size, device, quantile_indices, point_estimator, median_idx, predict_batch_fn
    )
    out = nan_row_fallback(out, flat)

    return unflatten_channel_independent(out, n_samples, n_features)
