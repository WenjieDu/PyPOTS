"""Regression tests for the standalone example runner's subprocess encoding."""

# License: BSD-3-Clause

import pytest

from scripts import run_all_examples


@pytest.mark.parametrize("exit_code", [0, 7])
def test_unicode_example_output(tmp_path, monkeypatch, exit_code):
    """Unicode survives a legacy inherited encoding, including failure diagnostics."""
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    script = tmp_path / "unicode_example.py"
    script.write_text(
        f"import sys\nprint('🚀 Training')\nprint('✅ 完成', file=sys.stderr)\nsys.exit({exit_code})\n",
        encoding="utf-8",
    )

    if exit_code == 0:
        run_all_examples.test_standalone_examples_can_run(script)
    else:
        with pytest.raises(AssertionError) as exc_info:
            run_all_examples.test_standalone_examples_can_run(script)
        message = str(exc_info.value)
        assert f"unicode_example.py exited with {exit_code}" in message
        assert "🚀 Training" in message
        assert "✅ 完成" in message
