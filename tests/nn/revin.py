"""Regression tests for RevIN's masked statistics (issue #968)."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch

from pypots.nn.modules.revin import RevIN


def test_masked_scale_is_the_standard_deviation_of_observed_values():
    torch.manual_seed(0)
    x = 10 + torch.randn(2, 200, 3)
    revin = RevIN(3, affine=False)
    revin(x, None, mode="norm")
    unmasked = revin.stdev.clone()
    revin(x, torch.ones_like(x), mode="norm")
    torch.testing.assert_close(revin.stdev, unmasked, rtol=1e-3, atol=1e-3)

    mask = torch.ones_like(x)
    mask[:, :50] = 0
    x_missing = x.masked_fill(mask == 0, 1e6)  # values at missing positions must not matter
    revin(x_missing, mask, mode="norm")
    observed = x[:, 50:]
    torch.testing.assert_close(revin.mean, observed.mean(1, keepdim=True), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(revin.stdev, observed.std(1, unbiased=False, keepdim=True), rtol=1e-3, atol=1e-3)
