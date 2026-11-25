<!-- Target the dev branch for ordinary contributions, not main. -->

## Summary

<!-- Describe what changed, why, and the affected APIs/tasks. Link a related issue or discussion when applicable. -->

## Validation

<!-- List the exact test commands, environment/device, and actual results. Identify tests not run and why. For model changes, include relevant correctness evidence, input/output contracts, and any missing-value handling or checkpoint compatibility considerations. -->

## Compatibility and licensing

<!-- Describe API/dependency changes, migration implications, optional dependencies, and the source/license of any reused code, data, or weights. For a new model, link the original paper and implementation, explain any adaptations, and confirm the inclusion/maintenance plan. -->

## Checklist

<!-- Mark applicable items; explain N/A in the sections above. Documentation-only changes need not run model tests. -->

- [ ] This PR targets `dev`, unless a maintainer has requested another base branch.
- [ ] The change is focused and relevant discussion or issue links are included where applicable.
- [ ] Scientific behavior and public API/data/checkpoint compatibility are preserved, or intentional changes and limitations are documented.
- [ ] Relevant tests have been added or updated and run; device-specific claims distinguish real hardware from mocked/CPU tests.
- [ ] Reused code, data, and weights have compatible licenses and required attribution; no confidential data or credentials are included.
- [ ] Relevant examples, documentation, and localized READMEs have been updated where applicable.
- [ ] `pre-commit run` (on staged changes), or `ruff check .` and `ruff format . --check`, pass locally.

See the [developer guide](https://docs.pypots.com/en/latest/dev_getting_started.html) and [testing guide](https://docs.pypots.com/en/latest/dev_testing.html).
