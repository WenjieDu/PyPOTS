NPU Support (Ascend)
========================

PyPOTS supports Huawei Ascend NPUs (e.g. Atlas A2 / Ascend 910B3) as a
device backend for compatible models. Select ``device="npu"`` or a specific
visible NPU with the standard model API for training, validation, prediction,
and checkpoint handling. Operator and model coverage is not identical to CUDA;
see the hardware notes below.

This page covers installation, single-device usage, multi-device DDP training,
offline loading of Hugging Face backbones, and hardware-specific limitations.
See :doc:`install` for installation from the development branch when the
installed release does not yet include NPU support.


Installation
----------------

PyPOTS does not add a hard dependency on torch-npu; CPU/CUDA users do not need it.
The following stack was previously verified on 8 × Ascend 910B3:

.. code-block:: text

    Python 3.12          CANN 9.0.0           torch 2.9.0+cpu
    torch_npu 2.9.0.post2                    driver 24.1.0.3

The rule of thumb: **torch_npu must match the torch version** (2.9.0 ↔ 2.9.0.post2), and CANN
ships with the container image (source ``set_env.sh`` from the CANN install path before
running anything, otherwise ``libhccl.so`` fails to load).

.. code-block:: bash

    # Install PyPOTS with NPU support (see Installation for source installation)
    pip install pypots
    # Example torch-npu version for the tested PyTorch 2.9.0 stack
    pip install torch-npu==2.9.0.post2
    # Source the set_env.sh script from your CANN installation before running Python.


Basic usage
----------------

.. code-block:: python

    import torch_npu  # register the NPU device before explicit selection
    from pypots.imputation import SAITS

    model = SAITS(
        n_steps=24, n_features=8,
        n_layers=1, d_model=32, n_heads=2,
        d_k=16, d_v=16, d_ffn=32,
        epochs=2, num_workers=0, device="npu",
    )

Use ``device="npu:3"`` to select a specific visible NPU, or ``device=None``
to probe CUDA first, then NPU, then CPU. Use the standard ``fit()``,
``predict()`` and ``save()`` / ``load()`` APIs. Checkpoint loading uses
``map_location`` to map tensors to the target device; cross-device loading
also requires a compatible model architecture and supported operators.


If you have existing CUDA scripts and want to run them on Ascend with zero code changes,
``torch_npu.contrib.transfer_to_npu`` (import it before any torch device call) monkey-patches
torch so that ``cuda`` silently maps to ``npu``. PyPOTS does not use this shim itself —
``device="npu"`` is a real first-class backend — but it is convenient for porting user code
that hard-codes ``device="cuda"``.


Offline loading of LLM backbones
----------------------------------------

TimeLLM, GPT4TS, and MOMENT download Hugging Face backbones (GPT-2, BERT, T5) on first use.
On air-gapped Ascend clusters, download the models on a machine with internet access, then
copy the cache directory over:

.. code-block:: bash

    # on the internet-facing machine
    huggingface-cli download openai-community/gpt2
    huggingface-cli download google-bert/bert-base-uncased
    huggingface-cli download t5-small

    # copy with symlinks resolved — the HF cache is a symlink farm and a plain
    # tar breaks it on extraction
    tar czf hf_cache.tar.gz --dereference -C ~/.cache/huggingface/hub .

    # on the Ascend host
    mkdir -p ~/.cache/huggingface/hub && tar xzf hf_cache.tar.gz -C ~/.cache/huggingface/hub

Then export ``HF_HUB_OFFLINE=1``. PyPOTS resolves the short model aliases (``gpt2``,
``bert-base-uncased``, ``t5-small`` …) to full repo ids (``openai-community/gpt2`` …) via
``pypots.utils.hf_offline.resolve_hf_repo_id``, because resolving a short alias normally
requires a Hub API call even when the model is fully cached — full repo ids read the local
cache directly.


Multi-device training (DDP)
----------------------------------

DDP on Ascend uses the HCCL backend registered by torch-npu. Use the
``run_ddp_fit()`` helper and launch one process per visible NPU with
``torchrun``. See :doc:`distributed` for the complete training script,
rank-local device selection, batch-size semantics and checkpoint handling.

.. code-block:: bash

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 PYPOTS_DEVICE=npu torchrun --standalone --nproc-per-node=4 train_ddp.py

This command uses the script in :doc:`distributed`. Use
``find_unused_parameters=True`` for models that require it, such as USGAN.

Previous real-device validation on 4 × Ascend 910B3 covered SAITS
(Transformer) and USGAN (unfused bidirectional GRU discriminator with
``find_unused_parameters=True``), which trained to completion with
identical best epochs across ranks. This is not a qualification of every
model, NPU type or software stack.


Hardware notes for Ascend 910B3
--------------------------------------

**Fused RNNs.** ``nn.GRU/LSTM`` over full sequences have no fp32 backward kernel on 910B3
(DynamicGRUV2); on some CANN versions even the fp32 forward fails graph build. PyPOTS
therefore substitutes mathematically equivalent *unfused* RNNs (step loops over basic ops,
``pypots/nn/modules/rnn_fallback.py``) on the NPU path. Parameter names follow the fused
layout (``weight_ih_l0`` etc.), so checkpoints are interchangeable between NPU and CPU/GPU.
fp16/AMP works too (rel_err ~6e-4); GRUCell/LSTMCell step loops (GRUD/BRITS/CSAI/CRLI style)
are fp32-safe as-is.

**Frequency-domain models.** ``torch.fft`` forward and backward work on NPU, but
complex64 matrix multiplies (matmul/einsum/addmm) are rejected by the Cube unit
(EZ1001). On the tested stack, training FiTS/FiLM/ETSformer/PatchTST/Crossformer/
MixLinear/GPVAE is limited by these operators. This limitation may vary with
the CANN and torch-npu versions.

**Integer ceil.** ``torch.ceil`` has no kernel for integer dtypes on NPU (EZ1001).
MOMENT's mask ratio code casts to float before ceiling (``pypots/nn/modules/moment/masking.py``).

**unfold backward.** The backward kernel for ``aten::unfold`` (aclnnUnfoldGrad) is
missing on NPU, so backprop through patching fails. MOMENT's ``Patching`` therefore uses a
mathematically equivalent gather-based unfold on NPU (``unfold_last_dim`` in
``pypots/nn/modules/moment/modules.py``); on CPU/CUDA the native unfold is kept.

**Virtualized hosts.** On some virtualized A2 hosts the HCCL gather family silently
corrupts data (``all_gather`` returns only rank-0 data, SMMU fault in plog, iBMA virtual
NICs without carrier). ``run_ddp_fit`` pre-broadcasts parameters and buffers, sets
``broadcast_buffers=False``, and on PyTorch versions exposing the ``init_sync``
option, skips DDP initialization-time synchronization. Validate
collective operations on your own host before relying on distributed results.

**num_workers.** Use ``num_workers=0`` in dataloaders on Ascend to avoid fork-related
deadlocks in HCCL initialization.
