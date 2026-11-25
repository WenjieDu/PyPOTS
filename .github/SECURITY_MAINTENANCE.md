# Dependency and supply-chain maintenance

This guide supplements [SECURITY.md](../SECURITY.md). It is for maintainers,
not a replacement for private vulnerability reporting.

## Repository controls (maintainer setup)

Review Settings → Security / Advanced Security and Rulesets on the repository.
The source workflow does not change repository settings, dismiss alerts, or
grant new tokens.

- Keep the dependency graph, Dependabot alerts and Dependabot security updates enabled.
- Enable secret scanning and push protection where available for this public repository.
- Keep private vulnerability reporting enabled; use the reporting policy in SECURITY.md.
- Review CodeQL default setup (Python and supported Actions analysis) and its results.
  Do not create a second competing CodeQL configuration if default setup is active.
- After hosted checks have passed, require the CI, lint and Dependency review checks
  for the relevant protected branches. Review the impact on existing branch rules first.
- Protect release tags and the default branch against unauthorized changes. Use
  restricted release environments and least-privilege publishing credentials.

The exact repository-side settings and private alert inventory must be verified
by a repository administrator. An unauthenticated public API cannot supply that
inventory. Review the actual dependency, affected range, direct/transitive path,
fix availability and impact on supported Python/PyTorch/device configurations
before deciding on each alert. Prioritize exploitable critical and high findings;
do not mark alerts dismissed solely because CI is green.

## Routine updates versus security updates

Dependabot checks Python manifests at the repository root and in requirements/,
GitHub Actions, and Docker build definitions weekly. Routine updates target dev.
Small development/documentation updates and minor/patch Action updates are grouped;
runtime/scientific dependency changes and major Action updates remain reviewable
individually. Docker base changes require image rebuild and platform validation.

GitHub security updates still target the default branch (main), regardless of the
routine update target. Review and synchronize any accepted fixes back to dev.
Do not enable unconditional auto-merge.

The Dependency review workflow runs on PRs targeting main or dev and fails for
new high/critical vulnerable dependencies reported by GitHub, including development
and unknown scopes. It is not a scanner for every existing vulnerability, nor a
replacement for dependency alert triage or scientific compatibility testing.
Dependency graph availability is a hosted prerequisite.

## Immutable Actions and CI lint tools

External GitHub Actions use the full commit SHA resolved from the upstream
version ref, with a version comment for human review and Dependabot. Verify the
upstream repository and tag-to-commit mapping before accepting changes. Review
permissions, changes to action dependencies, and privileged trigger behavior;
do not turn a read-only PR job into one that runs untrusted code with write access.

When changing lint dependencies, regenerate the hash-locked file for the Linux
x86_64 Python 3.13 lint job:

```bash
uv pip compile --python-version 3.13 --python-platform x86_64-unknown-linux-gnu \
    --only-binary :all: --generate-hashes requirements/requirements_lint.in \
    -o requirements/requirements_lint.txt
python -m pip install --only-binary=:all: --require-hashes -r requirements/requirements_lint.txt
```

Review all Ruff configuration and hook pins together when changing Ruff. For an
actionlint upgrade, obtain the versioned official release and checksum from the
upstream release, verify the download, and update the pinned checksum and URL
in linting.yml. Do not download a moving latest release at runtime.

References:
[Dependabot options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference),
[dependency review](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/about-dependency-review).


## Current security upgrade scope

The documentation environment and PyPI release-validation environment use
PyTorch 2.10.0 (Python 3.10, CPU), addressing the affected ranges of nine of
the ten PyTorch alerts reported for the former documentation pin, including
GHSA-53q9-r3pm-6pq6 and GHSA-63cw-57p8-fm3p. GHSA-rrmf-rvhw-rf47
(torch.jit.script, low severity) still affects 2.10.0; its first patched
version is 2.13.0. The documentation's PyG binary extensions require a
matching PyTorch CPU ABI. Reassess the remaining alert when matching wheels
are available, and verify Dependabot after deployment. An advisory with no
listed first patched version is not proof of a confirmed fix even when the
new pin lies outside its reported affected range.

Transformers 5.x migration (#933) and checkpoint serialization hardening
(#934) require separate compatibility work. Older runtime versions remain
allowed by the package manifest, and legacy CI baselines and the Docker
CUDA/PyTorch baseline are unchanged. Do not treat this targeted upgrade as
a complete remediation of all user runtime environments or arbitrary
checkpoint deserialization.
