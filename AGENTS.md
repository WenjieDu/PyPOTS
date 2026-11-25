# AGENTS.md — PyPOTS AI Agent Guide

## Project Overview and Boundaries

PyPOTS is a Python toolbox for machine learning on **Partially-Observed Time
Series** (POTS). It supports imputation, classification, forecasting, clustering,
anomaly detection, and representation learning on multivariate time series with
missing values.

- PyPOTS owns open-source models, model profiles and configuration, training,
  prediction, evaluation, and model-shaped H5 packaging.
- Expose reusable model functionality through the public Python APIs or
  `pypots-cli`. Keep task-specific orchestration out of shared model internals.
- Preserve existing public APIs, data contracts, result keys, and checkpoint
  semantics unless an explicitly requested change requires otherwise.

## Architecture and Integration Paths

Choose the integration path before implementing a model:

- **Standard NN:** reuse the task's `BaseNN{Task}` training and prediction
  infrastructure.
- **Complex NN:** use the same task contract but override training behavior only
  when the algorithm requires it (e.g., GANs, pretraining, multi-stage training).
- **Non-NN:** inherit the appropriate non-neural task base. Do not introduce
  neural-network wrappers or backbones unnecessarily (e.g., LOCF, Mean, Median).

Neural-network models normally follow the three-layer architecture:

```
pypots/nn/modules/{model}/       # Layer 1: pure PyTorch modules
pypots/{task}/{model}/core.py    # Layer 2: _ModelName(ModelCore), forward and criteria
pypots/{task}/{model}/model.py   # Layer 3: ModelName(BaseNN{Task}), public API
pypots/{task}/{model}/data.py    # Optional: DatasetForModelName(BaseDataset)
```

- Keep task wrappers, dataset loading, and CLI logic out of backbones. Reuse
  shared layers in `pypots.nn.modules` where appropriate.
- For standard NN cores, `forward()` returns a dict with the task-specific
  outputs and, when `calc_criterion=True`, `loss` during training or `metric`
  during validation. Follow the corresponding task base and reference model.
- Add `data.py` only when `BaseDataset` cannot satisfy the sample contract.
- Task directories: `imputation/`, `classification/`, `forecasting/`,
  `clustering/`, `anomaly_detection/`, and `representation/`.

## Model Implementation Conventions

- Use NumPy-style docstrings for public classes and methods.
- Naming: `BackboneModelName`, `_ModelName` (private core), `ModelName` (public
  API), and `DatasetForModelName` (custom dataset).
- For standard NN models, the core accepts `training_loss: Criterion` and
  `validation_metric: Criterion`. Preserve the existing `Criterion` sentinel
  behavior for falling back to `training_loss` during validation.
- Accept optimizer classes and instances. Follow the existing
  `isinstance(optimizer, Optimizer)` pattern, instantiate a class when needed,
  then call `self.optimizer.init_optimizer(self.model.parameters())`.
- Use class defaults or `None` for optimizer, loss, metric, and mutable container
  arguments as appropriate. Instantiate per model; never share a mutable default
  instance across models. Preserve explicitly supplied instances.
- Reuse standard data assembly, training, prediction, and checkpoint behavior.
  Custom training loops must preserve task contracts and relevant base behavior.
- For new models, use the task templates and matching reference implementation;
  register the public model in the task package imports and `__all__`, and add
  tests under `tests/{task}/{model}.py`.

## Code Quality

- Use Ruff for formatting and linting. Use the version pinned in
  `requirements/requirements_dev.txt`; keep CI and hook versions aligned.
- `pyproject.toml` is the source of truth for Ruff rules, 120-character line
  length, exclusions, and Python target version. Do not duplicate those settings.
- Run `ruff format .` and `ruff check .` (or `pypots-cli dev --lint_code`).
  For non-mutating verification, use `ruff format . --check` and `ruff check .`.
- Recommend `pre-commit install` once per clone. Use `pre-commit run` on staged
  changes, or `pre-commit run --all-files` for a full tracked-file check. Review
  any automatic changes and keep unrelated fixes out of the PR. Local hook
  installation is recommended; CI linting and formatting are the project standard.
- Do not weaken lint rules or bypass checks just to make validation pass.
- Review automated fixes for behavioral impact, especially unsafe fixes. Use
  narrowly scoped, justified `noqa` exceptions for compatibility or intentional
  behavior rather than blanket exclusions.

## Runtime Compatibility

- Maintain Python 3.10–3.14 support and the dependency combinations in
  `.github/workflows/testing_ci.yml`. Check the target branch's actual matrix.
- Valid Python syntax does not guarantee third-party runtime compatibility.
  Preserve compatible annotations for APIs that inspect types at runtime. For
  example, PyG `MessagePassing` may require `typing.Optional` or PyG type aliases
  instead of PEP 604 unions on Python versions before 3.14.
- Do not indiscriminately modernize runtime-inspected signatures, change public
  exceptions, or raise dependency minimums to resolve a local issue.
- Keep optional dependencies optional. Validate the oldest supported runtime
  affected by a change, not only the newest development environment.

## Device and Distributed Training

Apply these conventions where the target branch provides device and distributed
training support; do not assume development-only features exist in releases.

- Reuse `pypots/utils/devices.py` for device detection, normalization, and
  availability checks, and `pypots/utils/distributed.py` for DDP when available.
- Avoid hard-coded CUDA assumptions and preserve CPU and single-process behavior.
  Keep hardware-specific dependencies optional.
- Preserve rank-local device selection, data sharding, and synchronization of
  epoch metrics and early-stopping decisions across distributed ranks.
- Ensure checkpoint writes are rank-safe and save model states independently of
  parallel wrappers. Preserve existing single-process checkpoint compatibility.
- Distinguish real accelerator tests from mocked device checks and CPU DDP
  tests. Do not claim real hardware support from mocks alone.

## Testing and Verification

- Shared configuration: `tests/global_test_config.py` provides the DATA singleton,
  DEVICE, EPOCHS=2, paths, and datasets from
  `benchpots.datasets.preprocess_random_walk`.
- Generate shared test data before model tests with
  `python tests/global_test_config.py`.
- Use `@pytest.mark.xdist_group(name="{task}-{model}")` where tests share state.
  Standard model tests cover `test_0_fit`, `test_1_{task_verb}`,
  `test_2_parameters`, `test_3_saving_path`, and `test_4_lazy_loading`.
- Run targeted tests with `python -m pytest -rA tests/{task}/{model}.py`, or use
  `pypots-cli dev --run_tests -k "{model_name}"`.
- Cover both in-memory and HDF5 inputs where supported, public result keys,
  shapes, missingness behavior, and save/load contracts as applicable.
- Match verification scope to the change: model tests for model fixes; affected
  model regressions for shared modules and base classes; default-instance
  isolation tests (`tests/base/test_default_components.py`) for optimizer and
  criterion changes; device and multi-process tests under `tests/utils/` for
  hardware and DDP changes; matching `tests/cli/` modules for CLI changes (e.g.,
  `python -m pytest tests/cli/data_cmd.py`).
- For typing or dependency changes, verify affected runtime paths on the oldest
  relevant supported environment. For packaging changes, run `python -m build`.
- For broad changes, follow the full test workflow in
  `.github/workflows/testing_ci.yml`; run repository-wide lint and format checks.
- Report exact commands, actual environment, results, skipped tests, and known
  gaps. Do not claim that unrun tests or unavailable hardware passed.

## Documentation and Internationalization

- Keep public documentation consistent with actual behavior. Update relevant
  developer documentation and command examples when public CLI behavior changes.
- Use `docs/dev_getting_started.rst`, `docs/dev_standard_nn.rst`,
  `docs/dev_complex_nn.rst`, `docs/dev_non_nn.rst`, `docs/dev_testing.rst`, and
  task templates as the detailed integration and verification guides.
- For documentation changes, run `make -C docs html` with the Sphinx dependencies
  in `.readthedocs.yaml` and review new warnings and broken references.
- Synchronize relevant README changes across the English README and all nine
  localized READMEs under `i18n/`. Preserve technical meaning, command examples,
  identifiers, and URLs while respecting each language's style.
- Keep PyPOTS source license declarations aligned with the root `LICENSE`
  (BSD-3-Clause).

## Git Workflow

- Inspect the current branch, worktree, and pending changes before editing.
  Reuse the existing worktree unless there is a genuine isolation need.
- Target `dev` for ordinary pull requests unless the maintainer specifies another
  target. Do not assume the default branch contains unreleased features.
- Keep changes focused on the requested scope. Do not overwrite unrelated user
  changes or include private local files, secrets, generated data, or artifacts.
- Do not commit, push, merge, or delete branches without explicit authorization.
  Do not bypass hooks or force destructive Git operations without authorization.
- Report the changed files, verification results, and remaining limitations before
  requesting review.
