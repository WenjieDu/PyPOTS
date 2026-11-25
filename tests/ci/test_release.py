"""Regression tests for release identity and distribution validation."""

# License: BSD-3-Clause

import io
import subprocess
import tarfile
import zipfile

import pytest

from scripts import check_release


@pytest.fixture
def release_repo(tmp_path, monkeypatch):
    """Create a disposable Git repository; never modify PyPOTS refs."""

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Release test")
    git("config", "user.email", "release-test@example.invalid")
    git("commit", "--allow-empty", "-m", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "pypots").mkdir()
    (tmp_path / "pypots/version.py").write_text('__version__ = "2.1"\n', encoding="utf-8")
    git("add", "pypots/version.py")
    git("commit", "-m", "release")
    head = git("rev-parse", "HEAD")
    git("tag", "v2.1")
    git("update-ref", "refs/remotes/origin/main", head)
    monkeypatch.setattr(check_release, "ROOT", tmp_path)
    return git, base, head


def test_exact_stable_tag(release_repo):
    _, _, head = release_repo
    assert check_release.validate_checkout("v2.1", head) == ("2.1", head)
    assert check_release.validate_checkout("v2.1", "") == ("2.1", head)


@pytest.mark.parametrize("tag", ["v2.1rc1", "v2.1.dev0", "2.1", "v02.1", "v2.1/evil"])
def test_nonstable_or_invalid_tag(release_repo, tag):
    with pytest.raises(ValueError, match="Expected a stable"):
        check_release.validate_checkout(tag, "")


def test_mismatched_source_version(release_repo):
    git, _, head = release_repo
    git("tag", "v2.2")
    with pytest.raises(ValueError, match="does not match package version"):
        check_release.validate_checkout("v2.2", head)


def test_mismatched_event_commit(release_repo):
    _, base, _ = release_repo
    with pytest.raises(ValueError, match="same commit"):
        check_release.validate_checkout("v2.1", base)


def test_mismatched_checkout(release_repo):
    git, _, _ = release_repo
    git("commit", "--allow-empty", "-m", "later commit")
    with pytest.raises(ValueError, match="same commit"):
        check_release.validate_checkout("v2.1", "")


def test_tag_not_on_main(release_repo):
    git, base, head = release_repo
    git("update-ref", "refs/remotes/origin/main", base)
    with pytest.raises(subprocess.CalledProcessError):
        check_release.validate_checkout("v2.1", head)


def make_distributions(directory, wheel_version="2.1", source_version="2.1", name="pypots"):
    wheel = directory / "pypots-2.1-py3-none-any.whl"
    sdist = directory / "pypots-2.1.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pypots-2.1.dist-info/METADATA", f"Name: {name}\nVersion: {wheel_version}\n")
    with tarfile.open(sdist, "w:gz") as archive:
        data = f"Name: {name}\nVersion: {source_version}\n".encode()
        member = tarfile.TarInfo("pypots-2.1/PKG-INFO")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))


def test_matching_distributions(tmp_path):
    make_distributions(tmp_path)
    check_release.validate_distributions(tmp_path, "2.1")


@pytest.mark.parametrize(
    ("wheel_version", "source_version", "name"),
    [("2.2", "2.1", "pypots"), ("2.1", "2.2", "pypots"), ("2.1", "2.1", "other")],
)
def test_reject_wrong_distribution_metadata(tmp_path, wheel_version, source_version, name):
    make_distributions(tmp_path, wheel_version, source_version, name)
    with pytest.raises(ValueError, match="metadata does not match"):
        check_release.validate_distributions(tmp_path, "2.1")


def test_reject_extra_distribution(tmp_path):
    make_distributions(tmp_path)
    (tmp_path / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one wheel"):
        check_release.validate_distributions(tmp_path, "2.1")
