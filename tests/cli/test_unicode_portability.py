"""Regression tests for Unicode-safe PyPOTS CLI I/O."""

# License: BSD-3-Clause

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pypots.cli.base import safe_print
from pypots.cli.model import _save_yaml
from pypots.cli.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252", "gbk", "ascii"])
def test_safe_print_on_strict_streams(encoding):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding, errors="strict")
    safe_print("Unicode: 🚀 ✅ 完成 café", file=stream)
    stream.flush()
    output = raw.getvalue().decode(encoding)
    if encoding == "utf-8":
        assert "🚀 ✅ 完成 café" in output
    else:
        assert "\\U0001f680" in output
        assert "café" in output if encoding != "ascii" else "caf\\xe9" in output


def _run_cli(*args, encoding):
    env = os.environ.copy()
    env.update(PYTHONUTF8="0", PYTHONIOENCODING=encoding)
    return subprocess.run(
        [sys.executable, "-m", "pypots.cli.pypots_cli", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("encoding", ["cp1252", "gbk", "ascii"])
def test_data_describe_survives_limited_console_encoding(tmp_path, encoding):
    unicode_dir = tmp_path / "数据-🚀"
    unicode_dir.mkdir()
    csv_path = unicode_dir / "样本-✅.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,,6\n7,8,9\n", encoding="utf-8")

    result = _run_cli("data", "describe", "--input", str(csv_path), encoding=encoding)

    assert result.returncode == 0, result.stderr.decode(encoding, errors="replace")
    output = result.stdout.decode(encoding)
    assert "Per-feature missing rates:" in output
    if encoding in {"cp1252", "ascii"}:
        assert "\\u2588" in output or "\\u2591" in output


def test_recommend_writes_utf8_to_unicode_path_under_cp1252(tmp_path):
    output = tmp_path / "配置-✅.yaml"
    result = _run_cli(
        "recommend",
        "--task",
        "imputation",
        "--model",
        "SAITS",
        "--n_steps",
        "4",
        "--n_features",
        "2",
        "--n_samples",
        "10",
        "--output",
        str(output),
        encoding="cp1252",
    )

    assert result.returncode == 0, result.stderr.decode("cp1252", errors="replace")
    assert output.exists()
    assert "task: imputation" in output.read_text(encoding="utf-8")
    console = result.stdout.decode("cp1252")
    assert "Config saved to:" in console
    assert "\\u2705" in console


def test_load_config_accepts_utf8_bom_and_unicode(tmp_path):
    path = tmp_path / "配置.json"
    path.write_text(json.dumps({"name": "完成"}, ensure_ascii=False), encoding="utf-8-sig")
    assert load_config(str(path)) == {"name": "完成"}


def test_model_yaml_writer_uses_utf8(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "模型-✅.yaml"
    _save_yaml({"name": "完成 🚀"}, str(path))
    assert "完成 🚀" in path.read_text(encoding="utf-8")
