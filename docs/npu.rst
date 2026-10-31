NPU Support (Ascend)
========================

PyPOTS supports Ascend NPUs (e.g. Atlas A2 / Ascend 910B3) as a first-class device backend:
``device="npu"`` works in model initialization, training, validation, prediction, and
checkpoint saving/loading, at parity with ``device="cuda"``. No model code changes are needed.

This page covers installation, usage, multi-device DDP training, offline loading of the
LLM backbones used by TimeLLM/GPT4TS/MOMENT, and the hardware-specific notes you need
before training on Ascend.


Installation
----------------

PyPOTS itself does not add a new hard dependency: ``torch_npu`` is imported lazily, only on
the ``npu`` code path. To train on Ascend you need the following stack (verified on 8×910B3):

.. code-block:: text

    Python 3.12          CANN 9.0.0           torch 2.9.0+cpu
    torch_npu 2.9.0.post2                    driver 24.1.0.3

The rule of thumb: **torch_npu must match the torch version** (2.9.0 ↔ 2.9.0.post2), and CANN
ships with the container image (source ``set_env.sh`` from the CANN install path before
running anything, otherwise ``libhccl.so`` fails to load).

.. code-block:: bash

    pip install pypots            # no extra needed; torch_npu is your responsibility
    # on the Ascend host:
    pip install torch_npu==2.9.0.post2


Basic usage
----------------

.. code-block:: python

    from pypots.imputation import SAITS

    model = SAITS(n_steps=24, n_features=8, ..., device="npu")      # first visible NPU
    model = SAITS(..., device="npu:3")                              # a specific NPU
    model = SAITS(..., device=None)                                 # auto: cuda → npu → cpu

Training, validation, prediction, and checkpoint save/load all follow the standard PyPOTS
APIs — ``map_location`` handles cross-device checkpoint loading, so a checkpoint saved on
NPU loads on CPU/GPU and vice versa.

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

DDP on Ascend uses the HCCL backend (registered by ``torch_npu``) and is driven by
``torchrun``. PyPOTS ships a launcher that drives the model's own ``fit()`` inside the
process group — you do not rewrite the training loop:

.. code-block:: bash

    # one process per visible device; rank i maps onto the i-th *visible* device
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_script.py

.. code-block:: python

    # train_script.py
    from pypots.imputation import SAITS
    from pypots.utils.distributed import run_ddp_fit

    model = SAITS(n_steps=24, n_features=8, ..., device=None, epochs=10)
    run_ddp_fit(model, train_set, val_set)
    # GAN-style models whose discriminator is idle during generator steps:
    run_ddp_fit(model, train_set, val_set, find_unused_parameters=True)

The launcher:

1. moves the model to its rank-local device,
2. swaps the inner net for a DDP wrapper (parameters pre-broadcast from rank 0),
3. shards the train/val dataloaders per rank,
4. runs the model's own ``fit()`` unchanged (gradients all-reduce),
5. unwraps and loads the best state dict on every rank.

In a plain single-process call (no ``torchrun``), ``run_ddp_fit`` falls back to
``model.fit()`` — call sites need no branching.

Verified on 4×910B3: SAITS (Transformer) and USGAN (unfused biGRU discriminator,
``find_unused_parameters=True``) both train to completion with identical best epochs
across ranks; CPU hosts transparently use gloo instead of HCCL.


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
(EZ1001). FiTS/FiLM/ETSformer/PatchTST/Crossformer/MixLinear/GPVAE therefore cannot
train on NPU yet (tracked in P2).

**Integer ceil.** ``torch.ceil`` has no kernel for integer dtypes on NPU (EZ1001).
MOMENT's mask ratio code casts to float before ceiling (``pypots/nn/modules/moment/masking.py``).

**unfold backward.** The backward kernel for ``aten::unfold`` (aclnnUnfoldGrad) is
missing on NPU, so backprop through patching fails. MOMENT's ``Patching`` therefore uses a
mathematically equivalent gather-based unfold on NPU (``unfold_last_dim`` in
``pypots/nn/modules/moment/modules.py``); on CPU/CUDA the native unfold is kept.

**Virtualized hosts.** On some virtualized A2 hosts the HCCL gather family silently
corrupts data (``all_gather`` returns only rank-0 data, SMMU fault in plog, iBMA virtual
NICs without carrier). ``run_ddp_fit`` pre-broadcasts parameters and sets
``broadcast_buffers=False`` to skip DDP's gather-based shape verification; training itself
only relies on broadcast + all_reduce, which are data-exact on such hosts.

**num_workers.** Use ``num_workers=0`` in dataloaders on Ascend to avoid fork-related
deadlocks in HCCL initialization.
