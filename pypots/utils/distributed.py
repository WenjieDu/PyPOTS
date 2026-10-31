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

# Created by Jun Wang <jwangfx@connect.ust.hk>
# License: BSD-3-Clause

import contextlib
import inspect
import os
import sys
import time
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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

    # retry on the same port after a short backoff. Two transient rendezvous
    # failures are known, both observed as flaky CI failures:
    # 1. "Connection reset by peer" — a non-zero rank reaches TCPStore before
    #    the rank-0 server socket is ready (fresh mp.spawn/torchrun start).
    # 2. "Socket Timeout" — re-init right after destroy_process_group(): both
    #    ranks race to bind MASTER_PORT as the new store server and the loser
    #    times out (torch 2.3 era; also needs the retry on the SAME port).
    # The port must NOT change between attempts — the other ranks' stores
    # live there. Staggered backoff lets the winning server settle first.
    for attempt in range(5):
        try:
            dist.init_process_group(backend=backend)
            return True
        except Exception:
            if attempt == 4:
                raise
            if dist.is_initialized():
                dist.destroy_process_group()
            time.sleep(1.5 * (attempt + 1) + 0.5 * ddp_rank())
    return False


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
    skip_init_sync: bool = True,
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

    skip_init_sync :
        Skip DDP's initialization-time synchronization (parameter-shape
        verification across ranks, torch >= 2.2's ``init_sync``). That check
        uses the gather family of collectives, which silently corrupts data on
        some virtualized Ascend hosts (iBMA virtual NICs); pre_broadcast above
        already guarantees identical initial states, so the verification is
        redundant here. Parameter *structure* compatibility across ranks is
        still the caller's responsibility (same model class/config on every
        rank — which run_ddp_fit guarantees by construction).
    """
    if pre_broadcast and ddp_is_initialized() and device.type in ("cuda", "npu"):
        # only needed on gather-broken virtualized NPU hosts; on CPU/gloo the
        # plain broadcast of CPU tensors may lack a bound backend, and DDP's
        # own verification works fine there anyway
        _broadcast_module_states(module, src=0)
    kwargs = {"broadcast_buffers": False, "find_unused_parameters": find_unused_parameters}
    # torch >= 2.2: init_sync=False skips _verify_param_shape_across_processes
    # (gather-based) and _sync_module_states at construction; parameters were
    # pre-broadcast above instead. Older torch runs the check unconditionally
    # and works fine on healthy backends, so just don't pass the kwarg there.
    if skip_init_sync and "init_sync" in inspect.signature(DistributedDataParallel.__init__).parameters:
        kwargs["init_sync"] = False
    if device.type in ("cuda", "npu"):
        kwargs["device_ids"] = [device.index] if device.index is not None else None
    return DistributedDataParallel(module, **kwargs)


def shard_dataloader(dataloader: DataLoader, shuffle: bool = True, seed: int = 0) -> DataLoader:
    """Rebuild a DataLoader with a DistributedSampler so each rank sees its shard.

    Keeps the original dataset, batch size, and num_workers. Without this every
    rank trains on the full dataset and the all-reduced gradient is a stale sum
    instead of a mean over shards.
    """
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=_EpochShardedSampler(dataloader.dataset, shuffle=shuffle, seed=seed, drop_last=False),
        num_workers=dataloader.num_workers,
    )


class _EpochShardedSampler(DistributedSampler):
    """DistributedSampler that re-shuffles each epoch, driven by the epoch counter.

    The training loop counts epochs locally on the sharded loaders; without
    per-epoch re-shuffling every epoch would repeat epoch-1's shard order
    (vanilla DistributedSampler only re-seeds when set_epoch is called, and
    _train_model never calls it). The epoch counter increments on every
    __iter__, which reproduces DistributedSampler's documented contract
    ("each process will see a different sequence per epoch, unless set_epoch
    is called").

    on_epoch :
        Optional callback(epoch) fired at the start of every __iter__ — the
        test suite uses it to stamp which samples each rank consumes.
    """

    def __init__(self, *args, on_epoch=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_epoch = on_epoch

    def __iter__(self):
        self.epoch += 1
        self.set_epoch(self.epoch)
        if self._on_epoch is not None:
            self._on_epoch(self.epoch)
        return super().__iter__()


def all_reduce_mean(value: float) -> float:
    """All-reduce a python float across ranks and return the world mean."""
    t = torch.tensor([float(value)], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / ddp_world_size()


def _sync_epoch_metric(samplers: list, metric_fn) -> float:
    """Synchronize a per-epoch scalar metric across ranks.

    metric_fn() is called after every rank's local samplers have advanced to
    the same epoch, so the epoch counter itself is the rendezvous: ranks that
    run more local epochs would desync here — but the loop terminates on the
    synchronized decision below, keeping every rank in lockstep.
    """
    return all_reduce_mean(metric_fn())


@contextlib.contextmanager
def _sharded_dataloaders(on_sampler=None):
    """Make every DataLoader constructed inside this context rank-sharded.

    Model fit() implementations bind DataLoader at module import time
    (``from torch.utils.data import DataLoader``), so re-binding only
    torch.utils.data.DataLoader never reaches them. This patches the symbol
    in *every already-imported pypots module's namespace* plus the torch
    original, so loaders created by any model's fit() get a
    DistributedSampler injected (train/val alike, per-epoch reshuffling).
    Loaders constructed with an explicit sampler/batch_sampler are left alone.

    on_sampler :
        Optional callback(dataset) -> on_epoch callback (or None), invoked for
        every loader this context shards; the returned on_epoch(epoch) is fired
        at the start of each of that sampler's iterations — used by the test
        suite to audit which samples each rank consumes per epoch.
    """
    real_loader_cls = DataLoader

    def _make_sharded_loader_cls(rank):
        class _ShardingLoader:
            """Intercept DataLoader(...) to inject a rank-aware sampler."""

            def __new__(cls, dataset, *args, **kwargs):
                if kwargs.get("sampler") is None and kwargs.get("batch_sampler") is None:
                    kwargs.pop("shuffle", None)
                    on_epoch = on_sampler(dataset) if on_sampler is not None else None
                    kwargs["sampler"] = _EpochShardedSampler(
                        dataset,
                        shuffle=True,
                        seed=0,
                        drop_last=False,
                        on_epoch=on_epoch,
                    )
                return real_loader_cls(dataset, *args, **kwargs)

        return _ShardingLoader

    patched_namespaces = {}
    try:
        sharded_cls = _make_sharded_loader_cls(ddp_rank())
        # patch the torch source class plus every pypots module that imported it
        targets = [torch.utils.data]
        for mod_name, mod in list(sys.modules.items()):
            if mod is not None and mod_name.startswith("pypots.") and getattr(mod, "DataLoader", None) is real_loader_cls:
                targets.append(mod)
        for target in targets:
            patched_namespaces[target] = target.DataLoader
            target.DataLoader = sharded_cls
        yield
    finally:
        for target, original in patched_namespaces.items():
            target.DataLoader = original


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
      2. swaps model.model for a DDP wrapper (params pre-broadcast from rank 0,
         DDP's gather-based init verification skipped — see wrap_model_ddp),
      3. patches DataLoader construction so train/val loaders are sharded
         per rank with per-epoch reshuffling,
      4. calls fit() — _train_model runs unchanged except that the per-epoch
         selection metric is averaged across ranks, keeping early stopping,
         best_epoch and best_model_dict decisions identical on every rank,
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
    # in place, so it needs no rebuilding; only DataLoader sharding is patched.
    # The sharding context patches the DataLoader symbol in every already-
    # imported pypots module (models bind it at import time) so the sampler
    # actually reaches the loaders their fit() builds — train and val alike.
    try:
        with _sharded_dataloaders():
            model.fit(train_set, val_set, file_type=file_type)
    finally:
        model.model = original_net

    # the epoch metric was all-reduce-averaged inside _train_model, so the
    # best-epoch decision and best_model_dict are identical on every rank;
    # keys carry the DDP "module." prefix (deep-copied from the wrapper) —
    # strip it before loading into the plain net
    if model.best_model_dict is not None and any(k.startswith("module.") for k in model.best_model_dict):
        model.best_model_dict = {k.removeprefix("module."): v for k, v in model.best_model_dict.items()}
    model.model.load_state_dict(model.best_model_dict)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        logger.info(f"DDP training finished on {world} devices. Best model from epoch#{model.best_epoch} loaded.")
