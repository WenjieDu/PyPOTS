"""
Device backend utilities supporting CUDA, Ascend NPU, and CPU.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch

from .logging import logger

# device types PyPOTS understands, in probing order
DEVICE_TYPES = ("cuda", "npu", "cpu")


class _FallbackDevice:
    """Minimal torch.device stand-in for "npu"/"npu:i" strings when torch_npu is
    absent and torch itself refuses to construct the device. Only used so device
    *validation* can happen early with PyPOTS's clear error; never reaches .to().
    """

    def __init__(self, dev_type: str, index=None):
        self.type = dev_type
        self.index = index

    def __repr__(self):
        return f"device('{self.type}'" + (f", {self.index}" if self.index is not None else "") + ")"

    def __str__(self):
        return self.type + (f":{self.index}" if self.index is not None else "")

    def __eq__(self, other):
        return str(self) == str(other)


def _try_import_torch_npu():
    """Lazily import torch_npu; return the module or None (never raises).

    Importing torch_npu also registers the "npu" device type with torch, so
    torch.device("npu")/("npu:0") become constructible. When torch_npu is absent
    (e.g. plain CUDA/CPU environments), _register_npu_device_type below keeps
    device-string parsing working so users get PyPOTS's clear error instead of
    torch's "Expected one of cpu, cuda, ..." message.
    """
    try:
        import torch_npu  # noqa: F401

        return torch_npu
    except ImportError:
        return None


def _register_npu_device_type():
    """No-op kept for clarity: torch_npu registers the "npu" device type with torch
    on its own import, and we must NOT rename privateuse1 to "npu" ourselves when
    torch_npu is absent — that makes torch._C._accelerator_getAccelerator() report
    an npu accelerator with no torch.npu module behind it, and torch >= 2.10's
    optimizer.step() then crashes with "Device 'npu' does not have a corresponding
    module registered as 'torch.npu'" (verified on torch 2.14 cpu-only). Early
    string validation for "npu" without torch_npu is handled by
    parse_device_string()/resolve_device_type() instead.
    """
    _try_import_torch_npu()  # import for its registration side effect when present


def _npu_module():
    """Return torch_npu's device module (torch_npu.npu) or None."""
    torch_npu = _try_import_torch_npu()
    if torch_npu is None:
        return None
    return torch_npu.npu


def npu_is_available() -> bool:
    """Whether an Ascend NPU is usable in this environment."""
    mod = _npu_module()
    return mod is not None and mod.is_available() and mod.device_count() > 0


def cuda_is_available() -> bool:
    """Whether a CUDA GPU is usable in this environment."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def get_available_device_type() -> str:
    """Probe for an usable accelerator device type in the order cuda -> npu -> cpu.

    Returns
    -------
    device_type :
        One of "cuda", "npu", "cpu" — the first available one, "cpu" as the fallback.
    """
    if cuda_is_available():
        return "cuda"
    if npu_is_available():
        return "npu"
    return "cpu"


def resolve_device_type(device) -> str:
    """Normalize a user-given device spec (str / torch.device / list) to a device type.

    Raises
    ------
    ValueError
        If the spec is not a str/torch.device/list, or its type is not one of DEVICE_TYPES.
    """
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        lowered = device.lower()
        try:
            return torch.device(lowered).type
        except RuntimeError:
            # torch < 2.5 without torch_npu: parse "npu"/"npu:i" ourselves
            if lowered.split(":")[0] in DEVICE_TYPES:
                return lowered.split(":")[0]
            raise
    if isinstance(device, (list, tuple)):
        if len(device) == 0:
            raise ValueError("The list of devices should have at least 1 device, but got 0.")
        return resolve_device_type(device[0])
    raise TypeError(
        f"device should be str/torch.device/a list containing str or torch.device, but got {type(device)}"
    )


def parse_device_string(device_str: str) -> torch.device:
    """Construct a torch.device from a device string, accepting "npu"/"npu:i" even
    when torch_npu is absent (torch then raises "unknown device type" natively).

    When torch_npu is absent, a plain string-fallback object is still returned for
    "npu"/"npu:i" so PyPOTS can validate the device type and raise its clear
    "NPU is not available in your environment" assertion early — instead of letting
    torch's raw parse error leak. Callers must not .to(device) an npu device
    without checking availability first (check_device_availability does).
    """
    lowered = device_str.lower()
    try:
        return torch.device(lowered)
    except RuntimeError:
        dev_type = lowered.split(":")[0]
        if dev_type not in DEVICE_TYPES:
            raise
        # torch < 2.5 without torch_npu: hand-roll a device-like object
        # (real torch.device("npu") is impossible there — nothing is registered)
        index = lowered.split(":")[1] if ":" in lowered else None
        return _FallbackDevice(dev_type, index)


def device_module(device_type: str):
    """Return the torch device module for the given device type.

    Parameters
    ----------
    device_type :
        One of "cuda", "npu", "cpu". For "npu", torch_npu is lazily imported here;
        a RuntimeError with installation guidance is raised if it is missing.
    """
    if device_type == "cuda":
        return torch.cuda
    if device_type == "npu":
        mod = _npu_module()
        if mod is None:
            raise RuntimeError(
                "You are trying to use the Ascend NPU, but torch_npu is not installed in your environment. "
                "Please install the torch-npu wheel matching your torch version "
                "(e.g. torch 2.9.0 pairs with torch-npu 2.9.0.post2), "
                "and source the CANN environment script (set_env.sh) before running."
            )
        return mod
    if device_type == "cpu":
        return None  # torch has no torch.cpu module equivalent; callers handle cpu specially
    raise ValueError(f"Unknown device type: {device_type}. Should be one of {DEVICE_TYPES}.")


def check_device_availability(device_type: str) -> None:
    """Assert that the given device type is actually available, with a clear message."""
    if device_type == "cuda":
        assert cuda_is_available(), (
            "You are trying to use CUDA for model training, but CUDA is not available in your environment."
        )
    elif device_type == "npu":
        assert npu_is_available(), (
            "You are trying to use the Ascend NPU for model training, but NPU is not available in your environment. "
            "Please check that torch_npu is installed, the CANN environment script is sourced, "
            "and the device is visible (e.g. via npu-smi)."
        )
    # cpu is always available


def contains_full_sequence_rnn(module: torch.nn.Module) -> bool:
    """Whether the module contains a full-sequence nn.GRU / nn.LSTM (fused RNN).

    On Ascend NPU, the fused DynamicGRUV2 kernel lacks an fp32 backward pass,
    so models containing these layers cannot train in fp32 there (forward is fine).
    GRUCell/LSTMCell step loops are not affected — they compose base aten ops.
    """
    for m in module.modules():
        # isinstance covers both the classes and (via modules()) instances
        if isinstance(m, (torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN)):
            return True
    return False


def supports_device(device_type: str) -> bool:
    """Whether the given device type is one PyPOTS understands."""
    return device_type in DEVICE_TYPES
