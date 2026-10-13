"""
The core imputation logic of TS-ICL, ported from the (zero-shot, gradient-free)
``TSICL.impute()`` pipeline of the original implementation. Covariates are out of
scope here since PyPOTS' imputation contract has no notion of them. The checkpoint
loading, batching, and context/target grid plumbing shared with
``pypots.forecasting.tsicl`` live in ``pypots.nn.modules.tsicl``.

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
from einops import repeat
from torch import nn

from ...nn.modules.tsicl import (
    CustomStandardScaler,
    flatten_channel_independent,
    make_grid,
    nan_row_fallback,
    prepare_context_tensors,
    resolve_quantile_selection,
    run_batched_point_estimate,
    unflatten_channel_independent,
)


def _predict_batch(imputer: nn.Module, grid: torch.Tensor, series_c: torch.Tensor) -> torch.Tensor:
    """Run one forward pass of the TS-ICL imputer on a batch, in the model's normalized
    space; the whole grid is queried, and imputed values at originally-observed positions
    are overwritten with the (normalized) ground truth. Ported (covariate-free) from
    ``TSICL._get_context_target_coords_i`` + ``TSICL._predict_batch``.
    """
    context = prepare_context_tensors(grid=grid, series_c=series_c)
    series_c_ctx, coords_c = context["series_c"], context["coords_c"]
    coords_t = grid

    scaler = CustomStandardScaler(dim=1, epsilon=1e-5)
    scaler.fit(series_c_ctx)
    series_c_norm = scaler.transform(series_c_ctx)

    query_coords = torch.cat([coords_c, coords_t], dim=1)

    quantiles, _ = imputer(
        series=series_c_norm,
        coords=coords_c,
        target_coords=query_coords,
        undo_asinh_transform=True,
    )

    # positions in coords_t matching an (originally-observed) context coordinate are
    # forced back to the ground truth, so only genuinely missing positions are model-predicted
    match = coords_t == coords_c.transpose(1, 2)
    b_idx, t1_idx, t2_idx = match.nonzero(as_tuple=True)
    quantiles[b_idx, t1_idx] = repeat(series_c_norm[b_idx, t2_idx], "n 1 -> n q", q=quantiles.shape[-1])

    return scaler.inv_transform(quantiles)


def impute_with_tsicl(
    imputer: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    point_estimator: str = "median",
    quantile_levels: Optional[list[float]] = None,
) -> np.ndarray:
    """Impute a `(n_samples, n_steps, n_features)` array with NaNs, feature by feature
    (TS-ICL models one series at a time; see the channel-independence note in ``TSICL``'s
    docstring).
    """
    quantile_indices, median_idx, point_estimator = resolve_quantile_selection(
        imputer, point_estimator, quantile_levels
    )
    flat, n_samples, n_features = flatten_channel_independent(X)
    n_steps = flat.shape[1]
    imputer = imputer.to(device)

    def predict_batch_fn(series_c: torch.Tensor) -> torch.Tensor:
        grid = make_grid(n_steps, num_samples=series_c.shape[0]).to(device)
        return _predict_batch(imputer, grid, series_c)

    out = run_batched_point_estimate(
        flat, n_steps, batch_size, device, quantile_indices, point_estimator, median_idx, predict_batch_fn
    )
    out = nan_row_fallback(out, flat)

    # belt and suspenders: observed values are carried through bit-exact, regardless of how
    # faithfully the model's own replace_by_gt round-trip (z-normalize / denormalize) preserved them
    observed = np.isfinite(flat)
    out = np.where(observed, flat, out)

    return unflatten_channel_independent(out, n_samples, n_features)
