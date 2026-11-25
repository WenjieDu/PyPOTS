Distributed Training (DDP)
==========================

PyPOTS provides a training helper based on PyTorch DistributedDataParallel
(DDP) for compatible neural-network models. Each process trains a replica of
the model on its own data shard, and gradients are synchronized between ranks.
This is distinct from the existing CUDA DataParallel device-list interface.

Supported backends
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Device
     - Backend
     - Requirements
   * - NVIDIA CUDA GPU
     - NCCL
     - CUDA-enabled PyTorch and visible GPUs
   * - Ascend NPU
     - HCCL
     - Matching PyTorch, torch-npu, CANN and driver; see :doc:`npu`
   * - CPU
     - Gloo
     - PyTorch with distributed support

Use a PyPOTS version that contains ``pypots.utils.distributed.run_ddp_fit``.
See :doc:`install` for development-source installation if your release does
not include it. The examples below target a single host.

Quick start
-----------

Save the following script as ``train_ddp.py``. It uses synthetic time series,
so no dataset downloads are required. Every rank constructs the same model
and receives the same full dataset; PyPOTS shards the training and validation
DataLoaders internally.

.. code-block:: python

   import os

   import numpy as np

   backend = os.environ.get("PYPOTS_DEVICE", "cpu")
   if backend == "npu":
       import torch_npu  # registers the NPU device and HCCL backend

   from pypots.imputation import SAITS
   from pypots.utils.distributed import run_ddp_fit
   from pypots.utils.random import set_random_seed

   local_rank = int(os.environ.get("LOCAL_RANK", "0"))
   rank = int(os.environ.get("RANK", "0"))
   device = "cpu" if backend == "cpu" else f"{backend}:{local_rank}"

   set_random_seed(42)
   rng = np.random.default_rng(42)
   X = rng.normal(size=(64, 24, 8)).astype(np.float32)
   X_ori = X.copy()
   X[rng.random(X.shape) < 0.1] = np.nan

   train_set = {"X": X[:48]}
   val_set = {"X": X[48:], "X_ori": X_ori[48:]}

   model = SAITS(
       n_steps=24,
       n_features=8,
       n_layers=1,
       d_model=32,
       n_heads=2,
       d_k=16,
       d_v=16,
       d_ffn=32,
       epochs=2,
       batch_size=8,
       num_workers=0,
       device=device,
   )
   run_ddp_fit(model, train_set, val_set)

   if rank == 0:
       model.save("saits_ddp.pypots")
       print(model.impute({"X": X[:4]}).shape)

Launch the script using one process per visible accelerator:

.. code-block:: bash

   # Two NVIDIA GPUs
   CUDA_VISIBLE_DEVICES=0,1 PYPOTS_DEVICE=cuda torchrun --standalone --nproc-per-node=2 train_ddp.py

   # Two Ascend NPUs (initialize CANN first)
   ASCEND_RT_VISIBLE_DEVICES=0,1 PYPOTS_DEVICE=npu torchrun --standalone --nproc-per-node=2 train_ddp.py

   # Two CPU processes (also useful for local debugging)
   PYPOTS_DEVICE=cpu torchrun --standalone --nproc-per-node=2 train_ddp.py

``PYPOTS_DEVICE`` is an environment variable used by this example script, not
a built-in PyPOTS setting. The local rank indexes the *visible* devices.
Pass a single device to each model instance, not a device list. Without
``torchrun``, ``run_ddp_fit()`` falls back to the usual single-process
``model.fit()`` behavior.

How training works
------------------

The helper initializes the process group for the selected device type,
moves the model to its rank-local device, wraps its inner network in DDP,
and shards the DataLoaders created by its existing ``fit()`` method.
For compatible training loops, epoch metrics are synchronized to keep
early stopping and best-checkpoint selection consistent across ranks.
After successful training, the helper restores the plain network, loads
the best weights on each rank, and tears down the process group.

- ``batch_size`` is the batch size per rank. With two ranks and a local
  batch size of 8, a full distributed step uses 16 samples across ranks.
  Sampler padding and final batches can affect the effective sample count.
- Use identical model configurations, input datasets and compatible
  training control flow on all ranks. Do not pre-shard the dataset yourself.
- For models with parameters unused in some backward passes, such as the
  USGAN training pattern, pass ``find_unused_parameters=True`` to
  ``run_ddp_fit()`` when needed.
- The built-in automatic checkpoint saving is restricted to rank 0 during
  distributed training. After the helper returns, guard explicit saving,
  prediction and other single-writer operations with the original
  ``RANK`` environment variable, as shown above.

Compatibility and validation
----------------------------

DDP is intended for compatible neural-network training workflows, not every
model or arbitrary custom training loop. It does not automatically distribute
inference. Performance and scaling depend on the workload, local batch size,
communication backend and available hardware. Do not assume that DDP is faster
for every workload or that all models support every accelerator.

CPU/Gloo multi-process regression tests cover training, sampler sharding,
early stopping, initialization consistency and checkpoint handling.
The Ascend guide records previous real-device validation of SAITS and USGAN
on 4 × Ascend 910B3 and describes known operator limitations.
CUDA uses the NCCL backend; the CPU CI tests do not establish hardware
qualification for CUDA or Ascend.

For API details, see :mod:`pypots.utils.distributed`.
