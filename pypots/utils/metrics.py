"""
Utilities for evaluation metrics
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import torch


def cal_mae(inputs, target, mask=None):
    """ calculate Mean Absolute Error"""
    if mask is not None:
        return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)
    else:
        return torch.mean(torch.abs(inputs - target))


def cal_mse(inputs, target, mask=None):
    """ calculate Mean Square Error"""
    if mask is not None:
        return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)
    else:
        return torch.mean(torch.square(inputs - target))


def cal_rmse(inputs, target, mask=None):
    """ calculate Root Mean Square Error"""
    return torch.sqrt(cal_mse(inputs, target, mask))


def cal_mre(inputs, target, mask=None):
    """ calculate Mean Relative Error"""
    if mask is not None:
        return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(torch.abs(target * mask)) + 1e-9)
    else:
        return torch.mean(torch.abs(inputs - target)) / (torch.sum(torch.abs(target)) + 1e-9)
