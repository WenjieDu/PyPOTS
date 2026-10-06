# RFC: Ascend NPU Support for PyPOTS

- **RFC**: pypots-rfc-002 · **Status**: Draft · **Target**: PyPOTS 1.6 · **Date**: 2026-10-06
- **Hardware**: Atlas A2 (Ascend 910B3) · **Discussion**: GitHub Discussion

## 1. Motivation

PyPOTS 目前仅支持 NVIDIA GPU。本 RFC 提出让 Ascend NPU 成为一等设备后端：
`device="npu"` 在模型初始化、训练、验证、预测、存取点、CLI 上与 `device="cuda"` 同级可用。

范围：单卡（P0）→ AMP（P0，RNN 正确性前提，见 §2）→ 多卡 DDP（P1）。LLM 模型（TimeLLM/GPT4TS/MOMENT）放 P2。

**改造成本低的依据**：设备逻辑集中在 `pypots/base.py` 的 4 个方法，全仓仅 5 个文件触碰 `torch.cuda`，107 个模型零改动；无 triton/apex/自定义 CUDA 扩展依赖。

## 2. 硬件实证结论（910B3 实测，测试环境见 §5）

| 算子/模式 | fp32 | fp16/AMP |
|---|---|---|
| `nn.GRU/LSTM` 整序列 | ⚠️ **前向正常、反向失败**（DynamicGRUV2 无 fp32 backward kernel；前向实测 ~1.4 ms/batch 正常） | ✅ rel_err ~6e-4 |
| `Linear→GRU` 链 under autocast | — | ❌ fp16输入+fp32权重混合被拒（EZ3002；权重须 fp16） |
| `GRUCell/LSTMCell` 逐步循环（GRUD/BRTITS/CSAI 模式） | ✅ | ✅ |
| MHA / TransformerEncoder / LayerNorm / GELU / Dropout | ✅ | ✅ |
| Adam(foreach) / SGD / GradScaler / 掩码算术 | ✅ | ✅ |

要点：
- 用整序列融合 `nn.GRU/LSTM` 的是 **4 个模型：mRNN/SegRNN/StemGNN/USGAN**（grep 源码核实；CRLI 用 `GRUCell/LSTMCell` module-list 循环，fp32 安全）——它们在 NPU 上 fp32 **无法训练**，但推理/predict 可用。**对它们的训练，AMP 或等价 fallback 是正确性前提而非提速选项**
- 失败是异步的（fp32 backward kernel 缺失 → 炸在 `Adam.step`/`synchronize` 等不相关调用处），文档必须显式提示
- RNN fp16 精度损失实测 rel_err 6.1e-4，可接受；GRUD/BRTITS/CSAI/CRLI 用 Cell 循环，fp32 安全
- SegRNN 的融合 GRU 是**分块调用**（对 segment 后的 `(1, bc, d)`，非全长序列），cast 的成本/收益与其余三个不同

## 3. Design

**设备抽象**（新 `pypots/utils/devices.py`）：
- `get_available_device_type()` 探测顺序 cuda → npu → cpu；`resolve_device("npu:0")`
- torch_npu **lazy import**（仅 npu 路径触发），无新硬依赖；提供 `pypots[npu]` extra
- `_setup_device`（base.py）接受 `"npu"`/`"npu:0"`/list；多卡 npu 走 DDP，多卡 cuda 保持 DataParallel 不动

**RNN fp16 shim**（NPU 路径自动应用，`PYPOTS_NPU_RNN_FP16=0` 可关）：
- 检测模型含整序列 RNN 时，RNN 调用点前后做 fp32↔fp16 cast（含权重，loss 保持 fp32）
- 不改模型代码

**选项 C —— 等价 unfused fallback（按模型 opt-in，USGAN 已验证）**：
对 4 个融合 RNN 模型，把融合 `nn.GRU(bidirectional=True)` 替换为两个 `GRUCell` 步进循环（正序 + 逆序，输出拼接）。**数学等价**——同权重布局，输出与融合模块逐位一致（实测：CPU max err 0.0；NPU 1.5e-07，纯浮点顺序差）——只用基础算子（matmul/sigmoid/tanh），fp32 前向+反向在 NPU 全通，**零精度损失、不依赖 AMP**。
- 已端到端验证：USGAN 判别器经此替换在 910B3（CANN 9.0.0 容器内）训练成功（~22 s/epoch vs CPU ~156 s/epoch，约 7×）；repro + 等价性测试见 `jump-dispersion-project/contrib/pypots_npu_patch/`
- 与 shim 的权衡：每模型一次性改动（4 个，小且机械）；step-loop 的串行 dispatch 在融合 kernel 可用的设备（CUDA）上慢于融合 kernel——但无 fp16 rel_err（6.1e-4）、无 cast 机制、无异步失败面
- 建议形态：`layers.py` 内 device-adaptive 类（cuda/cpu 用融合、npu 用 Cell-loop），权重一致故 checkpoint 互通
- 注：SegRNN 的分块调用模式（§2）使其 fallback 改动量小于全长序列的三个

**AMP**：`pypots.nn.functional.autocast` 变 device-aware（`torch.amp.autocast("cuda")` ↔ `torch_npu.npu.amp.autocast`），GradScaler 同理，`ENABLE_AMP` 门控加 npu 探测。

**多卡 DDP（P1）**：新 `utils/distributed.py` launcher，HCCL 进程组 + torchrun 内部 spawn；`num_workers=0` 规避 fork 死锁；checkpoint 的 DDP unwrap/map_location 逻辑已有，无需改。

**其余**：`utils/random.py` 播种、`pypots-cli env/info` 探测、`tests/global_test_config.py` 加 npu；测试打 `@pytest.mark.npu` 标，默认跳过（`-m "not npu"`），CUDA/CPU CI 零影响。

**文档**：新增 NPU 页（安装、用法、shim 说明、DDP 配方）；`transfer_to_npu` 作为用户存量代码的零改动桥梁写入文档，不进框架代码。

**CI**：GitHub hosted 套件不变；NPU 套件跑在内网 A2 机器的 cann9.0.0 容器，流水线结果贴回 PR。

## 4. 分期

| Phase | 内容 |
|---|---|
| P0 | devices.py + `_setup_device` npu + device-aware autocast/GradScaler + RNN shim + 播种/CLI/测试 + SAITS/GRUD/BRTITS/Transformer 的 npu 测试 |
| P1 | DDP + HCCL 多卡 launcher + 2×910B3 冒烟 |
| P2 | LLM 模型；fp32 RNN 上游修复跟踪；性能调优 |

## 5. 版本矩阵

**唯一指定栈**（§2 全部实测结论基于它；A2 CANN 9.0.0 基础镜像容器自带）：

| Python | CANN | torch | torch_npu | 硬件 |
|---|---|---|---|---|
| **3.12** | **9.0.0** | **2.9.0+cpu** | **2.9.0.post2** | 8× Ascend 910B3（Atlas A2），驱动 24.1.0.3 |

配对规则：torch_npu 必须与 torch 同版本号（2.9.0↔2.9.0.post2）；CANN 随容器镜像携带。PyPOTS 自身维持 `requires-python >=3.9`（CI 测 3.9 + 3.11）——NPU 特性是增量能力，不抬基线。

同栈独立复现：USGAN（pypots 1.5）在 CANN 9.0.0 容器内 910B3 上，用 §3 选项 C 的判别器替换 patch 端到端训练成功——与融合 biGRU 的等价性 CPU 逐位一致（0.0）、NPU 1.5e-07；standalone repro 见 `jump-dispersion-project/contrib/pypots_npu_patch/`（等价性测试 + 2-epoch 训练冒烟，cpu 与 npu 双端跑通）。

## 6. Open questions

1. 融合 RNN 策略——现有三个选项：(a) 自动 fp16 shim 带 kill-switch，(b) opt-in shim，(c) 按模型等价 Cell-loop fallback（§3 选项 C，USGAN 已验证——零精度损失、不依赖 AMP）。(c) + 薄薄一层 device-adaptive 开关可能对 4 个受影响模型完全覆盖 (a)/(b)；shim 保留给含融合 RNN 的用户自定义模型兜底。
2. `device=None` 且无 cuda 时，npu 是否优先于 cpu？（倾向是）
3. 多卡 CUDA 是否也迁 DDP？（本 RFC 不动，另立 RFC）
4. DDP UX：`fit(device=["npu:0","npu:1"])` 内部 spawn torchrun，还是要求用户手动 torchrun？（倾向内部 spawn + 文档给手动路径）

## 7. 兼容性

CPU/CUDA 用户零行为变化（npu 仅 lazy 路径；默认探测仅在无 cuda 且有 npu 时不同）；checkpoint 格式不变（map_location 已支持跨设备）；Python ≥3.9 不变。
