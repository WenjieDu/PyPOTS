"""Regression tests for per-model default optimizer and loss instances."""

import pytest

from pypots import classification, forecasting
from pypots.nn.modules.brits.backbone import BackboneBRITS, BackboneRITS
from pypots.nn.modules.csai.backbone import BackboneBCSAI, BackboneCSAI
from pypots.nn.modules.micn.layers import MIC
from pypots.nn.modules.saits.loss import SaitsLoss
from pypots.optim import Adam


@pytest.mark.parametrize(
    "model_class,kwargs",
    [
        (
            classification.Autoformer,
            {
                "n_classes": 2,
                "n_layers": 1,
                "d_model": 16,
                "n_heads": 2,
                "d_ffn": 16,
                "factor": 3,
                "moving_avg_window_size": 3,
            },
        ),
        (
            classification.iTransformer,
            {"n_classes": 2, "n_layers": 1, "d_model": 16, "n_heads": 2, "d_k": 8, "d_v": 8, "d_ffn": 16},
        ),
        (
            classification.PatchTST,
            {
                "n_classes": 2,
                "patch_size": 8,
                "patch_stride": 8,
                "n_layers": 1,
                "d_model": 16,
                "n_heads": 2,
                "d_k": 8,
                "d_v": 8,
                "d_ffn": 16,
                "dropout": 0,
                "attn_dropout": 0,
            },
        ),
        (classification.TEFN, {"n_classes": 2, "n_fod": 4}),
        (
            classification.TimesNet,
            {"n_classes": 2, "n_layers": 1, "top_k": 2, "d_model": 16, "d_ffn": 16, "n_kernels": 3},
        ),
        (forecasting.DLinear, {"n_pred_steps": 2, "n_pred_features": 5, "moving_avg_window_size": 3, "d_model": 16}),
        (
            forecasting.FiLM,
            {
                "n_pred_steps": 2,
                "n_pred_features": 5,
                "window_size": [2],
                "multiscale": [1, 2],
                "modes1": 8,
                "d_model": 16,
            },
        ),
        (
            forecasting.MICN,
            {"n_pred_steps": 2, "n_pred_features": 5, "n_layers": 1, "d_model": 16, "conv_kernel": [2, 2]},
        ),
        (
            forecasting.ModernTCN,
            {
                "n_pred_steps": 2,
                "n_pred_features": 5,
                "patch_size": 3,
                "patch_stride": 2,
                "downsampling_ratio": 2,
                "ffn_ratio": 1,
                "num_blocks": [1],
                "large_size": [5],
                "small_size": [3],
                "dims": [16],
                "use_multi_scale": False,
            },
        ),
        (forecasting.SegRNN, {"n_pred_steps": 2, "n_pred_features": 5, "seg_len": 2, "d_model": 16}),
        (
            forecasting.TimesNet,
            {
                "n_pred_steps": 2,
                "n_pred_features": 5,
                "n_layers": 1,
                "top_k": 2,
                "d_model": 16,
                "d_ffn": 16,
                "n_kernels": 3,
            },
        ),
    ],
)
def test_default_optimizer_is_not_shared(model_class, kwargs):
    common = {"n_steps": 8, "n_features": 5, "epochs": 1, "device": "cpu", "verbose": False}
    first = model_class(**common, **kwargs)
    second = model_class(**common, **kwargs)

    assert isinstance(first.optimizer, Adam)
    assert isinstance(second.optimizer, Adam)
    assert first.optimizer is not second.optimizer
    assert first.optimizer.torch_optimizer is not second.optimizer.torch_optimizer
    for model in (first, second):
        actual = {id(p) for group in model.optimizer.torch_optimizer.param_groups for p in group["params"]}
        expected = {id(p) for p in model.model.parameters()}
        assert actual == expected

    custom = Adam(lr=0.005)
    third = model_class(**common, **kwargs, optimizer=custom)
    assert third.optimizer is custom


def test_default_loss_instances_are_not_shared():
    first = BackboneRITS(8, 5, 8)
    second = BackboneRITS(8, 5, 8)
    assert first.training_loss is not second.training_loss

    first = BackboneBRITS(8, 5, 8)
    second = BackboneBRITS(8, 5, 8)
    assert first.rits_f.training_loss is not second.rits_f.training_loss

    first = BackboneCSAI(8, 5, 8, 8)
    second = BackboneCSAI(8, 5, 8, 8)
    assert first.training_loss is not second.training_loss

    first = BackboneBCSAI(8, 5, 8, 8)
    second = BackboneBCSAI(8, 5, 8, 8)
    assert first.model_f.training_loss is not second.model_f.training_loss

    assert SaitsLoss(1, 1).loss_calc_func is not SaitsLoss(1, 1).loss_calc_func


def test_mic_default_kernel_lists_are_not_shared():
    first = MIC(feature_size=4)
    second = MIC(feature_size=4)
    assert first.conv_kernel == second.conv_kernel == [24]
    assert first.conv_kernel is not second.conv_kernel
