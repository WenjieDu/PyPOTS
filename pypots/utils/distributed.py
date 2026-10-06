"""
Distributed (multi-NPU / multi-GPU) training utilities for PyPOTS.

Adds DDP (distributed data parallel) training to PyPOTS models without touching
their single-device fit() implementations: run_ddp_fit() drives the model's own
fit() inside a torchrun-launched process group, with the inner net swapped for
a DDP wrapper and the dataloaders sharded per rank.

On Ascend NPUs the HCCL backend is used (registered by torch_npu); on NVIDIA
GPUs NCCL; both fall back to gloo on CPU-only hosts.

Usage (one process per visible device, launched by torchrun):
    torchrun --nproc_per_node=4 train_script.py
    # train_script.py:
    from pypots.utils.distributed import run_ddp_fit
    model = SAITS(..., device=None)
    run_ddp_fit(model, train_set, val_set)
    # GAN-style models whose discriminator is idle during generator steps:
    run_ddp_fit(model, train_set, val_set, find_unused_parameters=True)

On Ascend, restrict devices via ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 — torchrun's
local rank i maps onto the i-th *visible* device.

Virtualized-host safety: on some virtualized 910B3 hosts the gather family of
HCCL primitives silently corrupts data (all_gather returns only rank-0 data).
DDP's constructor verifies parameter shapes via gather, which fails there.
This module pre-broadcasts parameters from rank 0 and skips that verification;
training itself only uses broadcast + all_reduce, which are verified data-exact.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .logging import logger


def ddp_is_initialized() -> bool:
    """Whether this process is part of an initialized DDP process group."""
    return dist.is_available() and dist.is_initialized()


def ddp_rank() -> int:
    """Rank of this process in the DDP group; 0 in single-process runs."""
    return dist.get_rank() if ddp_is_initialized() else 0


def ddp_world_size() -> int:
    """World size of the DDP group; 1 in single-process runs."""
    return dist.get_world_size() if ddp_is_initialized() else 1


def init_distributed(backend: Optional[str] = None, device_type: Optional[str] = None) -> bool:
    """Initialize the DDP process group from torchrun/elastic environment variables.

    Returns True when this process is distributed (world size > 1 and the group
    got created). No-op returning False in plain single-process runs.

    backend :
        Explicit backend, or auto-detected when None — see device_type below.

    device_type :
        The device the model will train on ("npu" / "cuda" / "cpu"). Backend
        auto-detection keys off THIS, not off what the host happens to have:
        a CPU-training run on an NPU machine must use gloo, because hccl has no
        backend bound for CPU tensors ("No backend type associated with device
        type cpu"). None falls back to probing: npu→hccl, cuda→nccl, else gloo.
    """
    if not dist.is_available():
        return False
    if dist.is_initialized():
        return dist.get_world_size() > 1
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False

    if backend is None:
        if device_type is None:
            device_type = "cpu"
            try:
                import torch_npu

                if torch_npu.npu.is_available():
                    device_type = "npu"
            except ImportError:
                pass
            if torch.cuda.is_available():
                device_type = "cuda"
        backend = {"npu": "hccl", "cuda": "nccl"}.get(device_type, "gloo")

    dist.init_process_group(backend=backend)
    return True


def rank_local_device(model_device: torch.device) -> torch.device:
    """Device this rank should use: rank i maps to the i-th visible device.

    Only meaningful under torchrun (LOCAL_RANK set); single-process runs get
    the model's own device back.
    """
    if not ddp_is_initialized():
        return model_device
    local_rank = int(os.environ.get("LOCAL_RANK", ddp_rank()))
    return torch.device(f"{model_device.type}:{local_rank}")


def _broadcast_module_states(module: torch.nn.Module, src: int = 0) -> None:
    """Broadcast parameters and buffers from src to all ranks.

    Pre-syncing states lets DDP skip its gather-based shape verification, which
    fails on hosts where the HCCL gather family is broken (virtualized 910B3
    with iBMA virtual NICs: all_gather silently returns only rank-0 data).
    """
    with torch.no_grad():
        for param in module.parameters():
            dist.broadcast(param.data, src=src)
        for buf in module.buffers():
            dist.broadcast(buf.data, src=src)


def wrap_model_ddp(
    module: torch.nn.Module,
    device: torch.device,
    pre_broadcast: bool = True,
    find_unused_parameters: bool = False,
) -> DistributedDataParallel:
    """Wrap a module for DDP on the rank-local device.

    pre_broadcast :
        Sync parameters/buffers from rank 0 before wrapping. Necessary on
        gather-broken NPU hosts; harmless and cheap elsewhere — DDP would
        broadcast states itself anyway.

    find_unused_parameters :
        Forwarded to DDP. Set True for GAN-style models where some parameters
        (e.g. the discriminator during generator steps) do not contribute to
        every loss — otherwise DDP aborts with "Expected to have finished
        reduction in the prior iteration".
    """
    if pre_broadcast and ddp_is_initialized() and device.type in ("cuda", "npu"):
        # only needed on gather-broken virtualized NPU hosts; on CPU/gloo the
        # plain broadcast of CPU tensors may lack a bound backend, and DDP's
        # own verification works fine there anyway
        _broadcast_module_states(module, src=0)
    kwargs = dict(broadcast_buffers=False, find_unused_parameters=find_unused_parameters)
    if device.type in ("cuda", "npu"):
        kwargs["device_ids"] = [device.index] if device.index is not None else None
    return DistributedDataParallel(module, **kwargs)


def shard_dataloader(dataloader: DataLoader, shuffle: bool = True, seed: int = 0) -> DataLoader:
    """Rebuild a DataLoader with a DistributedSampler so each rank sees its shard.

    Keeps the original dataset, batch size, and num_workers. Without this every
    rank trains on the full dataset and the all-reduced gradient is a stale sum
    instead of a mean over shards.
    """
    from torch.utils.data.distributed import DistributedSampler

    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=DistributedSampler(dataloader.dataset, shuffle=shuffle, seed=seed, drop_last=False),
        num_workers=dataloader.num_workers,
    )


def _maybe_shard(loader: Optional[DataLoader]) -> Optional[DataLoader]:
    if loader is None:
        return None
    return shard_dataloader(loader)


def run_ddp_fit(
    model,
    train_set: Union[dict, str],
    val_set: Optional[Union[dict, str]] = None,
    file_type: str = "hdf5",
    find_unused_parameters: bool = False,
) -> None:
    """Train a PyPOTS model with DDP across all torchrun-launched ranks.

    Drives the model's own fit() inside the process group:
      1. moves the model to its rank-local device,
      2. swaps model.model for a DDP wrapper (params pre-broadcast from rank 0),
      3. patches DataLoader construction so train/val loaders are sharded,
      4. calls fit() — _train_model runs unchanged, gradients all-reduce,
      5. unwraps and loads the best state dict on every rank, tears the group down.

    In a plain single-process call (no torchrun), falls back to model.fit()
    so call sites need no branching.

    find_unused_parameters :
        Set True for GAN-style models (e.g. USGAN) where parts of the module
        (the discriminator during generator steps) do not contribute to every
        backward; otherwise DDP aborts with "Expected to have finished
        reduction in the prior iteration".
    """
    if not init_distributed(device_type=model.device.type if hasattr(model, "device") else None):
        model.fit(train_set, val_set, file_type=file_type)
        return

    rank, world = ddp_rank(), ddp_world_size()
    local_device = rank_local_device(model.device)

    # move the inner net to this rank's device, then wrap it for DDP
    model._setup_device(local_device)
    model._send_model_to_given_device()
    original_net = model.model
    ddp_net = wrap_model_ddp(
        original_net, local_device, pre_broadcast=True, find_unused_parameters=find_unused_parameters
    )
    model.model = ddp_net

    # the model's optimizer was bound to the same Parameter objects DDP wraps
    # in place, so it needs no rebuilding; only DataLoader sharding is patched
    real_loader_cls = DataLoader

    class _ShardingLoader:
        """Intercept DataLoader(...) construction inside fit() to inject the sampler."""

        def __new__(cls, dataset, *args, **kwargs):
            sampler = kwargs.pop("sampler", None)
            if sampler is None:
                from torch.utils.data.distributed import DistributedSampler

                kwargs["sampler"] = DistributedSampler(dataset, shuffle=kwargs.pop("shuffle", False), seed=0)
            return real_loader_cls(dataset, *args, **kwargs)

    import torch.utils.data as tud

    try:
        tud.DataLoader = _ShardingLoader
        model.fit(train_set, val_set, file_type=file_type)
    finally:
        tud.DataLoader = real_loader_cls
        model.model = original_net

    # every rank saw all-reduced epoch statistics, so best_model_dict is
    # identical across ranks; keys carry the DDP "module." prefix though —
    # best_model_dict was deep-copied from the wrapper, strip the prefix
    if model.best_model_dict is not None and any(k.startswith("module.") for k in model.best_model_dict):
        model.best_model_dict = {k.removeprefix("module."): v for k, v in model.best_model_dict.items()}
    model.model.load_state_dict(model.best_model_dict)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        logger.info(f"DDP training finished on {world} devices. Best model from epoch#{model.best_epoch} loaded.")
