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


def test_revin_masked_stats_ignore_unobserved_entries():
    x_obs = torch.tensor([[[8.0, 2.0], [10.0, 4.0], [12.0, 6.0]]])
    mask = torch.tensor([[[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])

    x_with_garbage = x_obs.clone()
    x_with_garbage[mask == 0] = 999.0

    revin_clean = RevIN(2, affine=False)
    revin_garbage = RevIN(2, affine=False)
    revin_clean(x_obs * mask, mask, mode="norm")
    norm_garbage = revin_garbage(x_with_garbage, mask, mode="norm")

    assert torch.allclose(revin_garbage.mean, revin_clean.mean, atol=1e-6)
    assert torch.allclose(revin_garbage.stdev, revin_clean.stdev, atol=1e-6)
    # Channel 0 observed values are [8, 12] -> mean=10, population std=2
    assert torch.isclose(revin_garbage.mean[0, 0, 0], torch.tensor(10.0), atol=1e-4)
    assert torch.isclose(revin_garbage.stdev[0, 0, 0], torch.tensor(2.0), atol=1e-4)

    restored = revin_garbage(norm_garbage, mode="denorm")
    assert torch.allclose(restored[mask == 1], x_with_garbage[mask == 1], atol=1e-5)


def test_revin_all_missing_mask_stays_finite():
    x = torch.ones(2, 8, 3) * 5.0
    mask = torch.zeros_like(x)
    revin = RevIN(3, affine=False)
    out = revin(x, mask, mode="norm")
    assert torch.isfinite(out).all()
