# RFC: Ascend NPU Support for PyPOTS

- **RFC**: pypots-rfc-002 · **Status**: Draft · **Target**: PyPOTS 1.6 · **Date**: 2026-10-06
- **Hardware**: Atlas A2 (Ascend 910B3) · **Discussion**: GitHub Discussion

## 1. Motivation

PyPOTS currently supports NVIDIA GPUs as its only accelerator backend. This RFC proposes making the Ascend NPU a first-class device backend: `device="npu"` works at the same level as `device="cuda"` — model init, training, validation, prediction, checkpointing, and the CLI.

Scope: single device (P0) → AMP (P0, a correctness prerequisite for RNN models, see §2) → multi-device DDP (P1). LLM-based models (TimeLLM/GPT4TS/MOMENT) are deferred to P2.

**Why the change is cheap**: device logic is centralized in 4 methods of `pypots/base.py`; only 5 files across the whole repo touch `torch.cuda`; all 107 model implementations inherit device behavior and need zero per-model changes; no triton/apex/custom CUDA extensions in the dependency tree.

## 2. Hardware findings (measured on 910B3; environments in §5)

| Op / pattern | fp32 | fp16 / AMP |
|---|---|---|
| Full-sequence `nn.GRU/LSTM` | ⚠️ **forward OK, backward fails** (DynamicGRUV2's fp32 backward kernel does not exist; fwd ~1.4 ms/batch runs fine) | ✅ rel_err ~6e-4 |
| `Linear→GRU` chain under autocast | — | ❌ fp16 input + fp32 weights rejected (EZ3002; weights must be fp16 too) |
| `GRUCell/LSTMCell` step loops (GRUD/BRTITS/CSAI pattern) | ✅ | ✅ |
| MHA / TransformerEncoder / LayerNorm / GELU / Dropout | ✅ | ✅ |
| Adam(foreach) / SGD / GradScaler / masked arithmetic | ✅ | ✅ |

Key points:
- 4 models use full-sequence fused `nn.GRU/LSTM` (**mRNN, SegRNN, StemGNN, USGAN** — verified by grep; CRLI uses `GRUCell/LSTMCell` module-list loops and is fp32-safe) — they **cannot train** on NPU in fp32, but inference/predict works. **For training them, AMP or an equivalent fallback is a correctness prerequisite, not a speed knob.**
- The failure is asynchronous (fp32 backward kernel missing → surfaces later in unrelated calls like `Adam.step`/`synchronize`) — documentation must call this out explicitly.
- Measured fp16 RNN accuracy cost: rel_err 6.1e-4 vs CPU fp32 — acceptable. GRUD/BRTITS/CSAI/CRLI use Cell loops and are fp32-safe.
- SegRNN calls its fused GRU **chunk-wise** (per-segment `(1, bc, d)`, not full length) — cast cost/benefit differs from the other three.

## 3. Design

**Device abstraction** (new `pypots/utils/devices.py`):
- `get_available_device_type()` probing cuda → npu → cpu; `resolve_device("npu:0")`
- torch_npu is **lazily imported** (only on npu paths) — no new hard dependency; an optional `pypots[npu]` extra will be provided
- `_setup_device` (base.py) accepts `"npu"` / `"npu:0"` / device lists; multi-device npu goes to DDP, multi-device cuda keeps DataParallel unchanged

**RNN fp16 shim** (applied automatically on npu paths; `PYPOTS_NPU_RNN_FP16=0` disables):
- When a model contains full-sequence RNNs, cast to fp16 around the RNN call site (weights included) and cast outputs back to fp32; loss computation stays fp32
- No model-code changes

**Option C — equivalent unfused fallback (per-model opt-in, already validated for USGAN)**:
For the 4 fused-RNN models, replace the fused `nn.GRU(bidirectional=True)` with two `GRUCell` step-loops (forward + reversed, outputs concatenated). This is **mathematically equivalent** — same weight layout, outputs bit-exact vs the fused module (measured: CPU max err 0.0; NPU 1.5e-07 float-ordering only) — uses only base ops (matmul/sigmoid/tanh), so fp32 fwd+bwd runs on NPU with **no precision loss and no AMP dependency**.
- Validated end-to-end: USGAN discriminator patched this way trains on 910B3 inside the CANN 9.0.0 container (~22 s/epoch vs ~156 s/epoch CPU, ~7×); repro + equivalence test at `jump-dispersion-project/contrib/pypots_npu_patch/`
- Trade-off vs the shim: one-time edit per model (4 models, small and mechanical) and a step-loop's sequential dispatch is slower than a fused kernel *when the fused kernel works* (i.e. on CUDA); but no fp16 rel_err (6.1e-4), no cast machinery, no async-failure surface
- Suggested shape: device-adaptive class in `layers.py` (fused on cuda/cpu, Cell-loop on npu), weights identical so checkpoints interoperate
- Note SegRNN's chunk-wise call pattern (§2) makes its fallback a smaller change than the full-sequence cases

**AMP**: `pypots.nn.functional.autocast` becomes device-aware (`torch.amp.autocast("cuda")` ↔ `torch_npu.npu.amp.autocast`), likewise GradScaler; the `ENABLE_AMP` gate learns npu availability.

**Multi-device DDP (P1)**: new `utils/distributed.py` launcher — HCCL process group + torchrun spawned internally; `num_workers=0` to avoid fork deadlocks; the existing DDP-unwrap / `map_location` checkpoint logic already handles saving and loading — no changes needed.

**The rest**: npu seeding in `utils/random.py`, NPU probing in `pypots-cli env/info`, npu DEVICE probe in `tests/global_test_config.py`; tests carry a `@pytest.mark.npu` marker and are skipped by default (`-m "not npu"`) — zero impact on CUDA/CPU CI.

**Docs**: a new NPU page (installation, usage, shim explanation, DDP recipe); `torch_npu.contrib.transfer_to_npu` is documented as a zero-code-change bridge for users' existing CUDA scripts — but is NOT used inside PyPOTS itself.

**CI**: the GitHub-hosted suite is unchanged; the NPU suite runs in a cann9.0.0 container on the intranet A2 machine, with pipeline results posted back to the PR.

## 4. Phases

| Phase | Contents |
|---|---|
| P0 | devices.py + npu path in `_setup_device` + device-aware autocast/GradScaler + RNN fp16 shim + seeding/CLI/tests + npu test coverage for SAITS/GRUD/BRTITS/Transformer |
| P1 | DDP + HCCL multi-device launcher + smoke test on 2×910B3 |
| P2 | LLM models; tracking the upstream fp32 RNN fix; performance tuning |

## 5. Version matrix

**The single specified stack** (all §2 findings measured on it; shipped in the A2 CANN 9.0.0 base-image container):

| Python | CANN | torch | torch_npu | Hardware |
|---|---|---|---|---|
| **3.12** | **9.0.0** | **2.9.0+cpu** | **2.9.0.post2** | 8× Ascend 910B3 (Atlas A2), driver 24.1.0.3 |

Pairing rule: torch_npu must match the torch version number (2.9.0 ↔ 2.9.0.post2); CANN ships with the container image. PyPOTS itself keeps `requires-python >=3.9` (CI tests 3.9 + 3.11) — the NPU feature is additive and does not raise the floor.

Independent reproduction on the same stack: USGAN (pypots 1.5) trains end-to-end in the CANN 9.0.0 container on 910B3 with the §3 Option-C discriminator patch — equivalence vs fused biGRU bit-exact on CPU (0.0) and 1.5e-07 on NPU; standalone repro at `jump-dispersion-project/contrib/pypots_npu_patch/` (equivalence test + 2-epoch train smoke, run on both cpu and npu).

## 6. Open questions

1. Fused-RNN strategy — three options now on the table: (a) auto fp16 shim with kill-switch, (b) opt-in shim, (c) per-model equivalent Cell-loop fallback (§3 Option C, validated for USGAN — no precision loss, no AMP dependency). (c) + a thin device-adaptive switch may subsume (a)/(b) for the 4 affected models; shim remains the fallback for user-defined models containing fused RNNs.
2. When `device=None` and no cuda is present, should npu take priority over cpu? (leaning yes)
3. Should multi-device CUDA also migrate to DDP? (out of scope here; separate RFC if desired)
4. DDP UX: `fit(device=["npu:0","npu:1"])` spawning torchrun internally, or requiring users to run torchrun themselves? (leaning internal spawn + documented manual path)

## 7. Compatibility

Zero behavior change for CPU/CUDA users (npu exists only on lazy paths; the default probe differs only when cuda is absent and npu is present); checkpoint format unchanged (`map_location` already supports cross-device loading); Python ≥3.9 unchanged.
