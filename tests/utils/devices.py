"""
Test cases for the functions and classes in module `pypots.utils.devices`.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk>
# License: BSD-3-Clause

import unittest

import pytest
import torch

from pypots.utils.devices import (
    DEVICE_TYPES,
    contains_full_sequence_rnn,
    get_available_device_type,
    npu_is_available,
    parse_device_string,
    resolve_device_type,
    supports_device,
)
from pypots.utils.random import set_random_seed

NPU_AVAILABLE = npu_is_available()


class TestDeviceHelpers(unittest.TestCase):
    def setUp(self):
        set_random_seed()

    def test_0_resolve_device_type(self):
        # str / torch.device / list all normalize to a device type
        assert resolve_device_type("cpu") == "cpu"
        assert resolve_device_type(torch.device("cpu")) == "cpu"
        assert resolve_device_type([torch.device("cpu")]) == "cpu"
        assert resolve_device_type(["cpu"]) == "cpu"
        # "npu" must normalize even when torch_npu is absent (string fallback)
        assert resolve_device_type("npu") == "npu"
        assert resolve_device_type("npu:0") == "npu"
        # unknown type raises (torch's own parse error for unknown device strings)
        with self.assertRaises(RuntimeError):
            resolve_device_type("unknown_device")
        with self.assertRaises(TypeError):
            resolve_device_type(123)

    def test_1_parse_device_string(self):
        dev = parse_device_string("cpu")
        assert dev.type == "cpu"
        # "npu:i" parses to type npu and the right index, with or without torch_npu
        dev = parse_device_string("npu:3")
        assert dev.type == "npu"
        assert str(dev).endswith(":3")
        with self.assertRaises(RuntimeError):
            parse_device_string("unknown_device")

    def test_2_get_available_device_type(self):
        dev_type = get_available_device_type()
        assert dev_type in DEVICE_TYPES
        # probing order is cuda -> npu -> cpu, cpu always as the fallback
        if not torch.cuda.is_available() and not NPU_AVAILABLE:
            assert dev_type == "cpu"

    def test_3_supports_device(self):
        for dev_type in DEVICE_TYPES:
            assert supports_device(dev_type)
        assert not supports_device("unknown_device")

    def test_4_contains_full_sequence_rnn(self):
        class WithFused(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.GRU(4, 4, batch_first=True)

        class WithCell(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = torch.nn.GRUCell(4, 4)

        assert contains_full_sequence_rnn(WithFused())
        assert not contains_full_sequence_rnn(WithCell())


# NPU-specific tests below: they exercise the real torch_npu path and are skipped
# automatically when no Ascend NPU is visible (e.g. the CPU-only CI runners).
@pytest.mark.npu
class TestNpuSpecific(unittest.TestCase):
    def setUp(self):
        if not NPU_AVAILABLE:
            self.skipTest("No Ascend NPU visible in this environment")
        set_random_seed()

    def test_0_device_roundtrip_on_npu(self):
        from pypots.utils.devices import check_device_availability, device_module

        assert get_available_device_type() in ("npu", "cuda")
        check_device_availability("npu")
        mod = device_module("npu")
        assert mod is not None and mod.device_count() > 0
        dev = parse_device_string("npu:0")
        assert dev.type == "npu" and dev.index == 0
        x = torch.randn(4, 4).to(dev)
        assert x.device.type == "npu"

    def test_1_unfused_rnn_equivalence_on_npu(self):
        from pypots.nn.modules.rnn_fallback import UnfusedGRU

        torch.manual_seed(2023)
        hidden, seq_len, n_in = 16, 7, 5
        x_npu = torch.randn(3, seq_len, n_in).to("npu")
        gru = UnfusedGRU(n_in, hidden, bidirectional=True, batch_first=True).to("npu")
        out_npu, h_n_npu = gru(x_npu)
        # the fused GRU on CPU is the numerical reference
        ref = torch.nn.GRU(n_in, hidden, bidirectional=True, batch_first=True)
        ref.load_state_dict(gru.state_dict())
        out_ref, h_n_ref = ref(x_npu.to("cpu"))
        torch.testing.assert_close(out_npu.cpu(), out_ref, rtol=1e-4, atol=1e-4)

    def test_2_unfused_rnn_backward_on_npu(self):
        from pypots.nn.modules.rnn_fallback import UnfusedGRU

        x = torch.randn(3, 7, 5, device="npu")
        gru = UnfusedGRU(5, 16, bidirectional=True, batch_first=True).to("npu")
        out, _ = gru(x)
        loss = out.sum()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gru.parameters())
