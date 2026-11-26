"""Validate an immutable release checkout and its distribution artifacts."""

# License: BSD-3-Clause

import argparse
import ast
import re
import subprocess
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_TAG = re.compile(
    r"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:\.(?:0|[1-9]\d*))?(?:(?:a|b|rc)(?:0|[1-9]\d*))?",
    flags=re.ASCII,
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_version():
    tree = ast.parse((ROOT / "pypots/version.py").read_text(encoding="utf-8"))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise ValueError("Could not find the literal package version")


def validate_checkout(tag, event_sha):
    if not RELEASE_TAG.fullmatch(tag):
        raise ValueError(f"Expected a stable or pre-release vX.Y[.Z][aN|bN|rcN] tag, got {tag!r}")
    head = git("rev-parse", "HEAD")
    tagged = git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    if head != tagged or (event_sha and head != event_sha):
        raise ValueError("Checkout, release tag and release event must identify the same commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", head, "refs/remotes/origin/main"], cwd=ROOT, check=True)
    version = source_version()
    if tag != f"v{version}":
        raise ValueError(f"Release tag {tag!r} does not match package version {version!r}")
    print(f"PASS: {tag} -> {head}, package version {version}, reachable from main", flush=True)
    return version, head


def distribution_metadata(dist):
    files = sorted(dist.iterdir())
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    if len(files) != 2 or len(wheels) != 1 or len(sdists) != 1 or not all(path.is_file() for path in files):
        raise ValueError("Expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as archive:
        (name,) = (name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        wheel_metadata = BytesParser().parsebytes(archive.read(name))
    with tarfile.open(sdists[0]) as archive:
        (member,) = (
            member
            for member in archive.getmembers()
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
        )
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Missing source distribution metadata")
        with stream:
            source_metadata = BytesParser().parsebytes(stream.read())
    return wheel_metadata, source_metadata


def validate_distributions(dist, version):
    for metadata in distribution_metadata(dist):
        if metadata["Name"] != "pypots" or metadata["Version"] != version:
            raise ValueError("Distribution metadata does not match the PyPOTS release")
    print(f"PASS: both distribution versions match {version}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--event-sha", default="")
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    version, head = validate_checkout(args.tag, args.event_sha)
    if args.dist is not None:
        validate_distributions(args.dist, version)
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"version={version}\ncommit={head}\n")


if __name__ == "__main__":
    main()
