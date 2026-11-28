"""Regression tests for TimeMixer and TimeMixer++ backbones with use_norm (issues #968 and #969)."""

# License: BSD-3-Clause

import pytest
import torch

from pypots.nn.modules.timemixer import BackboneTimeMixer
from pypots.nn.modules.timemixerpp import BackboneTimeMixerPP


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
