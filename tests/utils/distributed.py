"""
Test cases for the functions and classes in module `pypots.utils.distributed`.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk>
# License: BSD-3-Clause

import os
import socket
import sys
import unittest

import numpy as np
import pytest
import torch
import torch.distributed as dist
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

    # re-init a fresh group and check every rank selected the same best epoch
    assert init_distributed(device_type="cpu") is True
    best_epoch = torch.tensor([model.best_epoch], dtype=torch.float64)
    dist.all_reduce(best_epoch, op=dist.ReduceOp.SUM)
    assert int(best_epoch.item() / world) == model.best_epoch
    dist.destroy_process_group()


class TestDistributedMultiProcessCpu(unittest.TestCase):
    """End-to-end run_ddp_fit on CPU/gloo across spawned ranks (no torchrun needed).

    Skipped on Windows: torch's TCPStore hangs there in the spawned workers
    (the local torch 2.14/libuv store never becomes ready). CI runs Linux.
    """

    def setUp(self):
        set_random_seed()
        if sys.platform == "win32":
            self.skipTest("torch.distributed TCPStore hangs on Windows; run this on Linux/Unix")

    def test_0_run_ddp_fit_two_ranks(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        ctx = mp.get_context("spawn")
        mp.spawn(_cpu_ddp_worker, args=(2, port), nprocs=2, join=True)


# NPU-specific tests below: they exercise the real torch_npu path and are skipped
# automatically when no Ascend NPU is visible (e.g. the CPU-only CI runners).
# Multi-rank HCCL training is verified via torchrun — see the DDP recipe in
# docs/rfc/2026-10-06-npu-support.md; pytest itself always runs single-process.
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
