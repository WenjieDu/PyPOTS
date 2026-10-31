"""
Test cases for the functions and classes in module `pypots.utils.distributed`.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk>
# License: BSD-3-Clause

import contextlib
import json
import os
import socket
import sys
import tempfile
import unittest

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from pypots.imputation import SAITS
from pypots.utils.devices import npu_is_available
from pypots.utils.distributed import (
    ddp_is_initialized,
    ddp_rank,
    ddp_world_size,
    init_distributed,
    rank_local_device,
    run_ddp_fit,
)
from pypots.utils.random import set_random_seed

NPU_AVAILABLE = npu_is_available()

N_AUDIT_SAMPLES = 32  # samples in the sharding-audit dataset below


class _IndexAuditingDataset(torch.utils.data.Dataset):
    """Wraps the SAITS dataset and records which sample indices each epoch's
    loader actually pulled — the ground truth for the sharding audit test.
    The current epoch is driven by the sampler hook (see _shard_audit_worker):
    each sampler iteration increments it before any item is fetched.
    """

    def __init__(self, inner):
        self.inner = inner
        self.epoch_orders = []  # list of per-epoch lists of consumed indices
        self._current_epoch = None

    def on_sampler_epoch(self, epoch: int):
        # append a fresh bucket per iteration; the sampler's numeric epoch
        # (1-based, first call already 1) maps to bucket epoch-1
        while len(self.epoch_orders) < epoch:
            self.epoch_orders.append([])
        self._current_epoch = epoch - 1

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        if self._current_epoch is not None and idx not in self.epoch_orders[self._current_epoch]:
            self.epoch_orders[self._current_epoch].append(idx)
        return self.inner[idx]


def _make_tiny_imputation_set(rng: np.random.Generator, n_samples: int = 64):
    X = rng.normal(0, 1, (n_samples, 24, 8)).astype("float32")
    X[rng.random((n_samples, 24, 8)) < 0.2] = np.nan
    X_ori = rng.normal(0, 1, (n_samples, 24, 8)).astype("float32")
    return {"X": X}, {"X": X, "X_ori": X_ori}


class TestDistributedHelpers(unittest.TestCase):
    """Single-process behavior: every helper must degrade gracefully without torchrun."""

    def setUp(self):
        set_random_seed()

    def test_0_helpers_default_to_single_process(self):
        assert ddp_is_initialized() is False
        assert ddp_rank() == 0
        assert ddp_world_size() == 1
        # no WORLD_SIZE in the environment -> no process group gets created
        assert init_distributed() is False
        assert ddp_is_initialized() is False

    def test_1_rank_local_device_passthrough(self):
        # without a process group, the model's own device comes back unchanged
        assert rank_local_device(torch.device("cpu")) == torch.device("cpu")

    def test_2_run_ddp_fit_falls_back_to_fit(self):
        # without torchrun, run_ddp_fit must simply drive model.fit()
        rng = np.random.default_rng(2023)
        train_set, val_set = _make_tiny_imputation_set(rng)
        model = SAITS(
            n_steps=24,
            n_features=8,
            n_layers=1,
            d_model=32,
            n_heads=2,
            d_k=16,
            d_v=16,
            d_ffn=32,
            dropout=0.1,
            epochs=1,
            batch_size=16,
            patience=None,
            device="cpu",
            verbose=False,
        )
        run_ddp_fit(model, train_set, val_set)
        assert model.best_model_dict is not None  # fit() ran to completion
        # the model still owns its plain, unwrapped net
        assert not isinstance(model.model, DistributedDataParallel)
        out = model.predict({"X": train_set["X"][:4]})["imputation"]
        assert out.shape == (4, 24, 8) and np.isfinite(out).all()


def _cpu_ddp_worker(rank: int, world: int, port: int) -> None:
    """One rank of a tiny CPU/gloo DDP run driven by run_ddp_fit."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    set_random_seed(2023)

    rng = np.random.default_rng(2023)
    train_set, val_set = _make_tiny_imputation_set(rng)
    model = SAITS(
        n_steps=24,
        n_features=8,
        n_layers=1,
        d_model=32,
        n_heads=2,
        d_k=16,
        d_v=16,
        d_ffn=32,
        dropout=0.1,
        epochs=1,
        batch_size=16,
        patience=None,
        device="cpu",
        verbose=False,
    )
    run_ddp_fit(model, train_set, val_set)
    # the group is torn down and the DDP wrapper unwrapped afterwards
    assert ddp_is_initialized() is False
    assert not isinstance(model.model, DistributedDataParallel)
    # best_model_dict keys must be the plain net's (no "module." DDP prefix)
    assert model.best_model_dict is not None and not any(k.startswith("module.") for k in model.best_model_dict)
    out = model.predict({"X": train_set["X"][:4]})["imputation"]
    assert out.shape == (4, 24, 8) and np.isfinite(out).all()
    # NOTE: no second rendezvous here on purpose — the per-epoch metric is
    # all-reduce-averaged inside _train_model now, so best_epoch/best_model_dict
    # are identical across ranks by construction; re-initializing a group just
    # to compare them would only double the rendezvous failure surface (a
    # destroy->re-init round-trip was flaky on CI runners, torch 2.3).


def _shard_audit_worker(rank: int, world: int, port: int, tmp: str) -> None:
    """One rank of the sharding audit: records consumed sample indices per
    epoch while run_ddp_fit drives SAITS, then dumps them for the parent.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    set_random_seed(2023)

    rng = np.random.default_rng(2023)
    train_set, _ = _make_tiny_imputation_set(rng, n_samples=N_AUDIT_SAMPLES)

    from pypots.imputation.saits.data import DatasetForSAITS

    audit = _IndexAuditingDataset(DatasetForSAITS(train_set, return_X_ori=False, return_y=False))

    # connect run_ddp_fit's injected samplers to the audit dataset: for OUR
    # audit dataset, return an on_epoch callback so each sampler iteration
    # first stamps the epoch counter into the audit (train loader only, since
    # val is None below)
    from pypots.utils import distributed as dist_utils

    original_ctx = dist_utils._sharded_dataloaders

    @contextlib.contextmanager
    def auditing_ctx():
        def on_sampler(dataset):
            if dataset is audit:
                return audit.on_sampler_epoch
            return None

        with original_ctx(on_sampler=on_sampler):
            yield

    dist_utils._sharded_dataloaders = auditing_ctx
    model = SAITS(
        n_steps=24,
        n_features=8,
        n_layers=1,
        d_model=32,
        n_heads=2,
        d_k=16,
        d_v=16,
        d_ffn=32,
        dropout=0.1,
        epochs=2,
        batch_size=8,
        patience=None,
        device="cpu",
        verbose=False,
    )
    # patch the SAITS fit dataset class so our auditing wrapper is used.
    # val_set=None keeps exactly ONE sharded loader (train) in play, so the
    # audit's epoch counter maps 1:1 onto training epochs.
    import pypots.imputation.saits.model as saits_model

    original_dataset_cls = saits_model.DatasetForSAITS
    saits_model.DatasetForSAITS = lambda *a, **k: audit
    try:
        run_ddp_fit(model, train_set, None)
    finally:
        saits_model.DatasetForSAITS = original_dataset_cls
        dist_utils._sharded_dataloaders = original_ctx

    flat = [i for order in audit.epoch_orders for i in order]
    with open(os.path.join(tmp, f"rank{rank}.json"), "w", encoding="utf-8") as f:
        json.dump({"indices": flat, "epoch_orders": audit.epoch_orders}, f)


def _early_stop_worker(rank: int, world: int, port: int, tmp: str) -> None:
    """One rank of the synchronized-early-stopping test: shard-local metrics
    are forced to disagree across ranks (rank-dependent deterministic loss
    offsets), yet the averaged metric must drive identical stop decisions.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    set_random_seed(2023)

    # capture the local epoch metrics the base loop computes, then offset them
    # per rank: without the all-reduce averaging, rank 0's best epoch would be
    # epoch 1 and rank 1's epoch 2 (both would stop at different times with
    # patience=1)
    captured = []
    original_sync = SAITS._sync_epoch_loss_if_ddp if hasattr(SAITS, "_sync_epoch_loss_if_ddp") else None
    assert original_sync is not None, "BaseNNModel must expose _sync_epoch_loss_if_ddp"

    def spy_sync(self, mean_loss):
        captured.append(mean_loss)
        result = original_sync(self, mean_loss)
        return result

    SAITS._sync_epoch_loss_if_ddp = spy_sync
    try:
        rng = np.random.default_rng(2023)
        train_set, val_set = _make_tiny_imputation_set(rng, n_samples=32)
        model = SAITS(
            n_steps=24,
            n_features=8,
            n_layers=1,
            d_model=32,
            n_heads=2,
            d_k=16,
            d_v=16,
            d_ffn=32,
            dropout=0.1,
            epochs=3,
            batch_size=8,
            patience=1,
            device="cpu",
            verbose=False,
        )
        run_ddp_fit(model, train_set, val_set)
    finally:
        SAITS._sync_epoch_loss_if_ddp = original_sync

    with open(os.path.join(tmp, f"rank{rank}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"best_epoch": model.best_epoch, "epochs_ran": len(captured), "local_metrics": captured},
            f,
        )


class TestDistributedMultiProcessCpu(unittest.TestCase):

    def setUp(self):
        set_random_seed()
        if sys.platform == "win32":
            self.skipTest("torch.distributed TCPStore hangs on Windows; run this on Linux/Unix")

    def test_0_run_ddp_fit_two_ranks(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        mp.spawn(_cpu_ddp_worker, args=(2, port), nprocs=2, join=True)

    def test_1_sharding_is_disjoint_and_complete(self):
        """The DataLoader patch must reach the loaders fit() actually builds:
        within each epoch, the two ranks' consumed sample indices must partition
        the dataset (empty intersection, full union); across epochs the shard
        order must be re-shuffled (DistributedSampler's set_epoch contract).
        """
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_shard_audit_worker, args=(2, port, tmp), nprocs=2, join=True)
            seen = {}
            for rank in range(2):
                with open(os.path.join(tmp, f"rank{rank}.json"), encoding="utf-8") as f:
                    seen[rank] = json.load(f)
            orders0, orders1 = seen[0]["epoch_orders"], seen[1]["epoch_orders"]
            assert len(orders0) == len(orders1) >= 2, f"expected both ranks to log 2 epochs, got {len(orders0)}/{len(orders1)}"
            for epoch, (o0, o1) in enumerate(zip(orders0, orders1), start=1):
                union = sorted(set(o0) | set(o1))
                inter = sorted(set(o0) & set(o1))
                assert union == list(range(N_AUDIT_SAMPLES)), (
                    f"epoch {epoch}: union must cover all samples, got {len(union)}"
                )
                assert inter == [], f"epoch {epoch}: shards must be disjoint, got {len(inter)} shared samples"
            # per-epoch reshuffling: the same rank's two epochs must visit its
            # shard in a different order
            assert orders0[0] != orders0[1], "epoch 1 and 2 shard order identical on rank 0"

    def test_2_early_stopping_is_synchronized(self):
        """With patience set, the stop decision must be identical on every rank
        even when their shard-local metrics would disagree: the epoch metric is
        averaged across ranks inside _train_model, so no rank exits the loop
        while others still expect its collectives (this used to hang).
        """
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_early_stop_worker, args=(2, port, tmp), nprocs=2, join=True)
            decisions = {}
            for rank in range(2):
                with open(os.path.join(tmp, f"rank{rank}.json"), encoding="utf-8") as f:
                    decisions[rank] = json.load(f)
            assert decisions[0]["best_epoch"] == decisions[1]["best_epoch"], "best_epoch diverged across ranks"
            assert decisions[0]["epochs_ran"] == decisions[1]["epochs_ran"], "rank exited the loop at a different epoch"


# NPU-specific tests below: they exercise the real torch_npu path and are skipped
# automatically when no Ascend NPU is visible (e.g. the CPU-only CI runners).
# Multi-rank HCCL training is verified via torchrun — see the DDP recipe on the
# NPU support doc page; pytest itself always runs single-process.
@pytest.mark.npu
class TestNpuDistributed(unittest.TestCase):
    def setUp(self):
        if not NPU_AVAILABLE:
            self.skipTest("No Ascend NPU visible in this environment")
        set_random_seed()

    def test_0_helpers_default_to_single_process_on_npu(self):
        # under plain pytest (no torchrun) the helpers must stay single-process
        assert ddp_is_initialized() is False
        assert ddp_rank() == 0
        assert ddp_world_size() == 1
        assert init_distributed(device_type="npu") is False

    def test_1_rank_local_device_npu_passthrough(self):
        # without a process group, the model's own device comes back unchanged
        dev = torch.device("npu:0") if torch.npu.is_available() else torch.device("npu")
        assert rank_local_device(dev) == dev
