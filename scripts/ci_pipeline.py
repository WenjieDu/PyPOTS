"""Build, install and validate the current PyPOTS checkout in a clean CI environment."""

# License: BSD-3-Clause

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT, env=None):
    """Run a command and propagate its exit status."""
    print("+", *args, flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def pip(*args):
    run(sys.executable, "-m", "pip", *args)


def validate_artifacts(wheel, sdist):
    """Check the source and wheel metadata and license before installing."""
    # CI dependencies have been installed by this point. Keep bootstrap imports stdlib-only.
    from packaging.requirements import Requirement

    expected = {
        Requirement(line.strip())
        for line in (ROOT / "requirements/requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    license_bytes = (ROOT / "LICENSE").read_bytes()
    with zipfile.ZipFile(wheel) as archive:
        (metadata_name,) = (name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        prefix = metadata_name.removesuffix("METADATA")
        licenses = [name for name in archive.namelist() if name.startswith(prefix) and name.endswith("/LICENSE")]
        if not any(archive.read(name) == license_bytes for name in licenses):
            raise ValueError("Wheel is missing the repository license")
        if "pypots-cli = pypots.cli.pypots_cli:main" not in archive.read(prefix + "entry_points.txt").decode():
            raise ValueError("Wheel is missing the CLI entry point")
    with tarfile.open(sdist) as archive:
        (root,) = {member.name.split("/", 1)[0] for member in archive.getmembers()}
        source_metadata_file = archive.extractfile(root + "/PKG-INFO")
        source_license_file = archive.extractfile(root + "/LICENSE")
        if source_metadata_file is None or source_license_file is None:
            raise ValueError("Sdist metadata or license missing")
        source_metadata = BytesParser().parsebytes(source_metadata_file.read())
        if source_license_file.read() != license_bytes:
            raise ValueError("Sdist license differs from the repository license")
    for field in ("Name", "Version", "Requires-Python"):
        if metadata[field] != source_metadata[field] or not metadata[field]:
            raise ValueError(f"Wheel and sdist disagree on {field}")
    for distribution in (metadata, source_metadata):
        actual = {Requirement(value) for value in distribution.get_all("Requires-Dist", []) if "extra ==" not in value}
        if actual != expected:
            raise ValueError(
                f"Incorrect runtime dependency metadata: missing={expected - actual}, extra={actual - expected}"
            )
    print("PASS: wheel/sdist metadata, license, CLI entry point, runtime dependency declarations", flush=True)


def install(args):
    """Install a CPU torch wheel and the wheel built from this checkout."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch_version = args.torch if sys.platform == "darwin" else f"{args.torch}+cpu"
    constraints = output / "constraints.txt"
    constraints.write_text(f"torch=={torch_version}\nnumpy=={args.numpy}\npandas=={args.pandas}\n", encoding="utf-8")
    pip("install", "--upgrade", "pip")
    if sys.platform == "darwin":
        pip("install", "-c", str(constraints), f"torch=={torch_version}")
    else:
        # Download only torch from its specialized index. Resolve dependencies on PyPI.
        wheels = output / "torch-wheel"
        wheels.mkdir(exist_ok=True)
        pip(
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "--dest",
            str(wheels),
            f"torch=={torch_version}",
        )
        (torch_wheel,) = wheels.glob("torch-*.whl")
        pip("install", "-c", str(constraints), str(torch_wheel))
    pip("install", "-c", str(constraints), "-r", "requirements/requirements_ci.txt")
    # The existing developer CLI tests exercise sphinx-apidoc. This is not a docs-build CI.
    if args.full:
        pip(
            "install",
            "-c",
            str(constraints),
            "sphinx",
            "furo",
            "sphinxcontrib-bibtex",
            "sphinxcontrib-gtagjs",
            "sphinx-autodoc-typehints",
            "ruff==0.17.6",
        )
    if args.pyg:
        pip("install", "-c", str(constraints), "sentencepiece")
        pip(
            "install",
            "-c",
            str(constraints),
            "torch-geometric==2.8.1",
            "torch-scatter",
            "torch-sparse",
            "--only-binary=torch-scatter,torch-sparse",
            "-f",
            f"https://data.pyg.org/whl/torch-{args.torch}+cpu.html",
        )
    dist = output / "dist"
    build_args = [] if args.sdist else ["--wheel"]
    run(sys.executable, "-m", "build", *build_args, "--outdir", str(dist))
    (wheel,) = dist.glob("*.whl")
    if args.sdist:
        (sdist,) = dist.glob("*.tar.gz")
        run(sys.executable, "-m", "twine", "check", str(wheel), str(sdist))
        validate_artifacts(wheel, sdist)
    pip("install", "-c", str(constraints), str(wheel))
    pip("check")
    # Inspect the installed package from outside the checkout, including CPU build identity.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    run(
        sys.executable,
        "-c",
        "import importlib.metadata as m, pathlib, torch, pypots; "
        "assert torch.version.cuda is None and torch.version.hip is None; "
        "assert pathlib.Path(pypots.__file__).resolve().is_relative_to(pathlib.Path(m.distribution('pypots').locate_file('')).resolve()); "
        "assert pypots.__version__ == m.version('pypots'); "
        "print('PyPOTS', pypots.__version__, pypots.__file__, 'PyTorch', torch.__version__)",
        cwd=output,
        env=env,
    )
    pip("list")


def examples(args):
    """Execute examples outside the repository, against the installed package."""
    output = args.output.resolve()
    sandbox = output / f"examples-{args.scope}"
    sandbox.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "examples", sandbox / "examples", dirs_exist_ok=True)
    (sandbox / "scripts").mkdir(exist_ok=True)
    runner = sandbox / "scripts" / "run_all_examples.py"
    shutil.copy2(ROOT / "scripts" / "run_all_examples.py", runner)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    run(
        sys.executable,
        str(runner),
        "--scope",
        args.scope,
        "--workers",
        "1",
        "--junitxml",
        str(output / f"examples-{args.scope}.xml"),
        cwd=sandbox,
        env=env,
    )


def package(args):
    """Verify the actual wheel and sdist, not just a source-tree import."""
    output = args.output.resolve()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    executable = shutil.which("pypots-cli")
    if executable is None:
        raise RuntimeError("Installed pypots-cli entry point not found")
    run(executable, "--help", cwd=output, env=env)
    args.scope = "smoke"
    examples(args)
    # Reinstall from the sdist; verify its build and entry point independently.
    (sdist,) = (output / "dist").glob("*.tar.gz")
    pip("install", "--force-reinstall", "--no-deps", str(sdist))
    pip("check")
    run(executable, "--help", cwd=output, env=env)
    run(
        sys.executable,
        "-c",
        "import importlib.metadata as m, pypots; assert pypots.__version__ == m.version('pypots'); "
        "print(pypots.__file__)",
        cwd=output,
        env=env,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("install", "examples", "package"):
        child = sub.add_parser(command)
        child.add_argument("--output", required=True, type=Path)
        if command == "install":
            child.add_argument("--torch", required=True)
            child.add_argument("--numpy", required=True)
            child.add_argument("--pandas", required=True)
            child.add_argument("--pyg", action="store_true")
            child.add_argument("--full", action="store_true")
            child.add_argument("--sdist", action="store_true")
        if command == "examples":
            child.add_argument("--scope", choices=("smoke", "offline", "network", "all"), required=True)
    args = parser.parse_args()
    {"install": install, "examples": examples, "package": package}[args.command](args)


if __name__ == "__main__":
    main()
