"""
This module provides functions for gathering data.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import torch


def gather_listed_dicts(dict_list: list) -> dict:
    """Gather batched dict output from model forward

    Parameters
    ----------
    dict_list:
        A list of dict output from model forward. Each dict should have the same keys.

    Returns
    -------
    gathered_dict:
        A dict with the same keys as the input dict, but with values concatenated along the batch dimension.

    """

    # check if all dicts have the same keys
    keys = dict_list[0].keys()
    for d in dict_list[1:]:
        assert d.keys() == keys, "All dicts should have the same keys"

    gathered_dict = dict()
    for k in keys:
        if isinstance(dict_list[0][k], torch.Tensor):
            if dict_list[0][k].dim() > 0:
                gathered_dict[k] = torch.cat([d[k] for d in dict_list], dim=0).cpu().detach().numpy()
        elif isinstance(dict_list[0][k], np.ndarray):
            if dict_list[0][k].ndim > 0:
                gathered_dict[k] = np.concatenate([d[k] for d in dict_list], axis=0)
        else:
            raise ValueError("Only support torch.Tensor and np.ndarray")

    return gathered_dict
