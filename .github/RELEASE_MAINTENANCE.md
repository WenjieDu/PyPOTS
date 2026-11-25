# Release maintenance

The PyPI and DockerHub workflows publish only for a published, non-draft,
non-prerelease GitHub Release. Pushing a release, release_pypi or release_docker
branch no longer publishes anything. Manual workflow dispatch validates an
existing stable tag, without publishing or logging into a registry. Workflows
and their validation scripts must already be present in the target release tag.

## One-time configuration before deploying

1. In PyPI, manage the existing pypots project → Publishing → Add a GitHub
   Trusted Publisher with repository owner WenjieDu, repository PyPOTS,
   workflow filename publish_to_PyPI.yml, and environment pypi.
   See [PyPI's setup instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
2. Create/review the GitHub environment named pypi. Restrict deployments to
   protected release tags and configure appropriate maintainer approval.
   Protect release tags from unauthorized creation and movement.
3. Keep the existing PYPI_API_TOKEN secret until Trusted Publishing has been
   independently confirmed by a successful intended release. The new workflow
   does not read it; do not delete it or silently fall back to token publication.
4. Confirm DOCKERHUB_USERNAME and DOCKERHUB_TOKEN are configured for the
   existing DockerHub repository. Use a scoped write token, never a personal
   password. No registry credentials are used in manual validation runs.
5. Verify both GitHub-hosted runner architectures and review the remaining
   dependency alerts before authorizing a release. Do not re-publish an old tag
   simply to test credentials.

## Release sequence

1. Merge accepted changes into main. Set the stable package version in
   pypots/version.py in a reviewed commit. Do not publish a development,
   alpha, beta or release-candidate version through these workflows.
2. Create the matching stable tag (vX.Y or vX.Y.Z) on that commit. The release
   validation checks the exact tag commit, the release event SHA, reachability
   from main, and the literal version in source. Do not move release tags.
3. From the default branch, run both publishing workflows manually with this
   tag if pre-publication validation is desired. Review the package artifact
   checks and both image smoke checks. This validation cannot publish.
4. Publish the GitHub Release for the exact tag. The workflows build from the
   release event's commit, not the current main branch.
5. Review both publishing results and the registry artifacts. Serialize actual
   releases; do not publish old releases out of order or overlap them.

## PyPI

The unprivileged validation job uses Python 3.10, PyTorch 2.10.0 CPU,
NumPy 1.26.4 and pandas 2.3.3. It builds a wheel and sdist once for publication,
checks their metadata, license, entry point and declared requirements, installs
the wheel, runs six installed-package smoke examples, checks the sdist install
and validates version metadata against the tag. It uploads only the two
validated distributions and their SHA256 manifest.

The separate publishing job downloads that exact artifact, checks integrity, and
uses the pinned PyPA publishing Action with short-lived OIDC credentials. It has
id-token: write, no source checkout, and no build. It requires a configured
Trusted Publisher and the pypi environment. It cannot run on a manual dispatch.

## DockerHub

Both architectures build on native Linux runners from the checked-out source
using the existing NVIDIA CUDA 12.1/cuDNN 8 Ubuntu 22.04 development base,
pinned by its multi-platform digest. There is no Git clone inside the Dockerfile.
The build context contains only the package inputs.

| Platform | PyTorch distribution | Optional graph extensions | Existing tags | Additional version tags |
| --- | --- | --- | --- | --- |
| linux/amd64 | 2.5.1+cu121, CUDA 12.1 | torch-scatter and torch-sparse binary wheels | latest, ubuntu-amd64, ubuntu-x86_64 | X.Y[.Z], X.Y[.Z]-amd64 |
| linux/arm64 | 2.5.1, CPU only | torch-geometric; scatter/sparse not promised | ubuntu-arm64 | X.Y[.Z]-arm64 |

The ARM64 image retains the historical CUDA base but explicitly installs a CPU
PyTorch build; it does not claim ARM64 CUDA acceleration. The amd64 image is a
CUDA-enabled build, but hosted CPU smoke tests cannot prove GPU execution. No
Ascend NPU image is published. The existing latest tag remains amd64, not a new
multi-architecture manifest. Existing aliases remain available.

Each native job builds and locally loads its image, checks architecture,
revision, installed package version, PyTorch build, a CPU tensor operation,
pip check, CLI help, and the graph extensions promised for that architecture, then pushes that same tested image under the
appropriate tags. Registry login is after validation. OCI labels record the
release version and source revision. The platform jobs are independent: one
platform may publish successfully while the other fails. PyPI and DockerHub
publication are also not an atomic transaction.

The pinned CUDA base and Torch/NumPy compatibility baseline are intentionally
retained; review base-image and dependency security alerts separately before
release. Changing the CUDA/PyTorch matrix requires compatibility evidence.
The native amd64 runner must have enough space for the CUDA development base
and PyTorch layers; check actual hosted results rather than assuming capacity.

## Failure recovery

- Before publishing: correct the source or environment and rerun manual
  validation. Do not change a published tag to different code.
- PyPI: a published version cannot be overwritten. Confirm which artifacts
  were uploaded before retrying a failed job. If the version is incorrect,
  follow the PyPI yank/replacement-release procedure; publish a new version.
- DockerHub: retain versioned tags; if a platform or alias failed, inspect the
  published digest and OCI labels. Retry only the appropriate failed job for
  the same release when safe. Restore a previous known-good digest to moving
  tags if necessary, rather than rebuilding from a moving branch.
- A successful source or local test is not proof of PyPI OIDC, real GPU
  execution, DockerHub authorization, or hosted publication. Do not publish
  live artifacts as a test of the workflow.
