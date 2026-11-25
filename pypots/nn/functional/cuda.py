""" """

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch


def autocast(device_type: str = None, enabled: bool = True, **kwargs):
    """Device-aware autocast wrapper, supporting CUDA and Ascend NPU.

   Keeps the historical signature ``autocast(enabled=...)`` working for CUDA (torch >=2.4
    routes through ``torch.amp.autocast("cuda")``; older versions use ``torch.cuda.amp``),
    and adds NPU via ``torch_npu.npu.amp.autocast`` when the device type is "npu".

    Parameters
    ----------
    device_type :
        "cuda", "npu", or None. When None, the device type of the current default device
        is used if it is cuda/npu, otherwise "cuda" (preserving the historical behavior).

    enabled :
        Whether autocast is enabled, default as True.

    """
    if device_type is None:
        # infer from the default device if it is an accelerator, else keep historical cuda
        default_type = torch.get_default_device().type if hasattr(torch, "get_default_device") else "cpu"
        device_type = default_type if default_type in ("cuda", "npu") else "cuda"

    if device_type == "npu":
        try:
            import torch_npu
        except ImportError as e:
            raise RuntimeError(
                "NPU autocast requires torch_npu. Please install the torch-npu wheel matching your torch "
                "version and source the CANN environment script."
            ) from e
        return torch_npu.npu.amp.autocast(enabled=enabled, **kwargs)

    if torch.__version__ < "2.4":
        from torch.cuda.amp import autocast

        return autocast(enabled=enabled, **kwargs)
    else:
        from torch.amp import autocast

        return autocast("cuda", enabled=enabled, **kwargs)
