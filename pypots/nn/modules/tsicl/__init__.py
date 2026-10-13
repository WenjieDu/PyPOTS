"""
The TS-ICL model architecture (Perceiver encoder + in-context-learning transformer head),
for use by ``pypots.imputation.tsicl`` and ``pypots.forecasting.tsicl``.

Notes
-----
Official pretrained weights hosted on Hugging Face are subject to the TS-ICL model license.

"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause

from .encoder import PerceiverEncoder
from .icl_learning import ICLearning, ICLearningCrossAttn
from .inference_utils import (
    HF_REPO_ID,
    CustomStandardScaler,
    build_tsicl_network,
    complete_nans,
    fetch_X,
    flatten_channel_independent,
    get_quantile_indices,
    load_tsicl_checkpoint,
    make_grid,
    nan_row_fallback,
    prepare_context_tensors,
    resolve_inference_device,
    resolve_quantile_selection,
    run_batched_point_estimate,
    unflatten_channel_independent,
)
from .network import TSICLNetwork

__all__ = [
    "HF_REPO_ID",
    "CustomStandardScaler",
    "ICLearning",
    "ICLearningCrossAttn",
    "PerceiverEncoder",
    "TSICLNetwork",
    "build_tsicl_network",
    "complete_nans",
    "fetch_X",
    "flatten_channel_independent",
    "get_quantile_indices",
    "load_tsicl_checkpoint",
    "make_grid",
    "nan_row_fallback",
    "prepare_context_tensors",
    "resolve_inference_device",
    "resolve_quantile_selection",
    "run_batched_point_estimate",
    "unflatten_channel_independent",
]
