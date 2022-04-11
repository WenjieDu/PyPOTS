"""
Utilities for evaluation metrics
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import torch
import numpy as np


def cal_mae(inputs, target, mask=None):
    """ calculate Mean Absolute Error"""
    lib = np if isinstance(inputs, np.ndarray) else torch
    if mask is not None:
        return lib.sum(lib.abs(inputs - target) * mask) / (lib.sum(mask) + 1e-9)
    else:
        return lib.mean(lib.abs(inputs - target))


def cal_mse(inputs, target, mask=None):
    """ calculate Mean Square Error"""
    lib = np if isinstance(inputs, np.ndarray) else torch
    if mask is not None:
        return lib.sum(lib.square(inputs - target) * mask) / (lib.sum(mask) + 1e-9)
    else:
        return lib.mean(lib.square(inputs - target))


def cal_rmse(inputs, target, mask=None):
    """ calculate Root Mean Square Error"""
    lib = np if isinstance(inputs, np.ndarray) else torch
    return lib.sqrt(cal_mse(inputs, target, mask))


def cal_mre(inputs, target, mask=None):
    """ calculate Mean Relative Error"""
    lib = np if isinstance(inputs, np.ndarray) else torch
    if mask is not None:
        return lib.sum(lib.abs(inputs - target) * mask) / (lib.sum(lib.abs(target * mask)) + 1e-9)
    else:
        return lib.mean(lib.abs(inputs - target)) / (lib.sum(lib.abs(target)) + 1e-9)
