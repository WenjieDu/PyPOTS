"""Regression tests for RevIN masked standard deviation and TimeMixer/TimeMixer++ use_norm calls."""

import pytest
import torch

from pypots.nn.modules.revin import RevIN
from pypots.nn.modules.timemixer import BackboneTimeMixer
from pypots.nn.modules.timemixerpp import BackboneTimeMixerPP


def test_revin_masked_scale_matches_unmasked_stdev():
    torch.manual_seed(0)
    x = 10.0 + torch.randn(2, 32, 3)
    all_ones_mask = torch.ones_like(x)

    revin_unmasked = RevIN(3, affine=False)
    norm_unmasked = revin_unmasked(x, mode="norm")

    revin_masked = RevIN(3, affine=False)
    norm_masked = revin_masked(x, all_ones_mask, mode="norm")

    assert torch.allclose(revin_masked.mean, revin_unmasked.mean, atol=1e-4)
    assert torch.allclose(revin_masked.stdev, revin_unmasked.stdev, atol=1e-4)
    assert torch.allclose(norm_masked, norm_unmasked, atol=1e-4)

    restored = revin_masked(norm_masked, mode="denorm")
    assert torch.allclose(restored, x, atol=1e-5)


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


@pytest.mark.parametrize("use_x_mark", [False, True])
def test_timemixer_backbone_forecast_and_anomaly_detection_with_use_norm(use_x_mark):
    backbone = BackboneTimeMixer(
        task_name="long_term_forecast",
        n_steps=8,
        n_features=3,
        n_pred_steps=4,
        n_pred_features=3,
        n_layers=1,
        d_model=8,
        d_ffn=16,
        dropout=0.0,
        top_k=2,
        channel_independence=True,
        decomp_method="moving_avg",
        moving_avg=3,
        downsampling_layers=1,
        downsampling_window=2,
        downsampling_method="avg",
        use_future_temporal_feature=False,
        use_norm=True,
        embed="timeF",
        freq="h",
    )
    x = 10.0 + torch.randn(2, 8, 3)
    x_mark = torch.zeros(2, 8, 4) if use_x_mark else None
    out = backbone.forecast(x, x_mark)
    assert out.shape == (2, 4, 3)
    assert torch.isfinite(out).all()

    anomaly_backbone = BackboneTimeMixer(
        task_name="anomaly_detection",
        n_steps=8,
        n_features=3,
        n_pred_steps=8,
        n_pred_features=3,
        n_layers=1,
        d_model=8,
        d_ffn=16,
        dropout=0.0,
        top_k=2,
        channel_independence=True,
        decomp_method="moving_avg",
        moving_avg=3,
        downsampling_layers=1,
        downsampling_window=2,
        downsampling_method="avg",
        use_future_temporal_feature=False,
        use_norm=True,
        embed="timeF",
        freq="h",
    )
    anomaly_calls = []
    orig_forward = anomaly_backbone.normalize_layers[0].forward

    def spy_forward(*args, **kwargs):
        anomaly_calls.append(kwargs.get("mode"))
        return orig_forward(*args, **kwargs)

    anomaly_backbone.normalize_layers[0].forward = spy_forward
    dec_out = anomaly_backbone.anomaly_detection(x)
    assert dec_out.shape == (2, 8, 3)
    assert torch.isfinite(dec_out).all()
    assert anomaly_calls == ["norm", "denorm"]


@pytest.mark.parametrize("use_x_mark", [False, True])
def test_timemixerpp_backbone_forecast_and_anomaly_detection_with_use_norm(use_x_mark):
    backbone = BackboneTimeMixerPP(
        task_name="long_term_forecast",
        n_steps=8,
        n_features=3,
        n_pred_steps=4,
        n_pred_features=3,
        n_layers=1,
        d_model=8,
        d_ffn=16,
        n_heads=2,
        dropout=0.0,
        top_k=2,
        n_kernels=2,
        channel_mixing=False,
        channel_independence=True,
        downsampling_layers=1,
        downsampling_window=2,
        downsampling_method="avg",
        use_future_temporal_feature=False,
        use_norm=True,
        embed="timeF",
        freq="h",
    )
    x = 10.0 + torch.randn(2, 8, 3)
    x_mark = torch.zeros(2, 8, 4) if use_x_mark else None
    out = backbone.forecast(x, x_mark)
    assert out.shape == (2, 4, 3)
    assert torch.isfinite(out).all()

    anomaly_backbone = BackboneTimeMixerPP(
        task_name="anomaly_detection",
        n_steps=8,
        n_features=3,
        n_pred_steps=8,
        n_pred_features=3,
        n_layers=1,
        d_model=8,
        d_ffn=16,
        n_heads=2,
        dropout=0.0,
        top_k=2,
        n_kernels=2,
        channel_mixing=False,
        channel_independence=True,
        downsampling_layers=1,
        downsampling_window=2,
        downsampling_method="avg",
        use_future_temporal_feature=False,
        use_norm=True,
        embed="timeF",
        freq="h",
    )
    anomaly_calls = []
    orig_forward = anomaly_backbone.revin_layers[0].forward

    def spy_forward(*args, **kwargs):
        anomaly_calls.append(kwargs.get("mode"))
        return orig_forward(*args, **kwargs)

    anomaly_backbone.c_out = 3
    anomaly_backbone.revin_layers[0].forward = spy_forward
    dec_out = anomaly_backbone.anomaly_detection(x)
    assert dec_out.shape == (2, 8, 3)
    assert torch.isfinite(dec_out).all()
    assert anomaly_calls == ["norm", "denorm"]
