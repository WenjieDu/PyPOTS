"""Check structural documentation contracts without building Sphinx or accessing the network."""

# License: BSD-3-Clause

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("IMPT", "FCST", "CLAF", "CLUS", "ANOD", "REPR")
TASK_DIRS = ("imputation", "forecasting", "classification", "clustering", "anomaly_detection", "representation")
EXAMPLE_URL = re.compile(r"https://github\.com/WenjieDu/PyPOTS/blob/main/(examples/[^\s)>]+\.py)")


def model_name(value):
    """Remove markup while retaining the model identity and adaptation marker."""
    value = re.sub(r"\[\^\d+\]", "", value)
    value = re.sub(r":cite:`[^`]+`", "", value)
    value = re.sub(r"<[^>]+>", "", value)
    return value.replace("|tsa-logo|", "").replace("`", "").replace("__", "").strip()


def normalize_task_cell(task, cell, name, source):
    """Validate a task marker and its example link."""
    links = EXAMPLE_URL.findall(cell)
    marked = "✅" in cell
    if links:
        if len(links) != 1 or not marked or not links[0].startswith(f"examples/{task}/"):
            raise ValueError(f"{source}: invalid {name} / {task} example link: {cell}")
        if not (ROOT / links[0]).is_file():
            raise ValueError(f"{source}: missing {links[0]}")
    elif marked and name != "Time-Series.AI":
        raise ValueError(f"{source}: {name} / {task} is marked but has no example")
    elif cell and not marked:
        raise ValueError(f"{source}: unexpected {name} / {task} cell: {cell}")
    return marked, tuple(links)


def normalize_model_row(row, source):
    """Validate the model row shape and normalize its six task cells."""
    if len(row) != 9:
        raise ValueError(f"{source}: expected 9 columns, got {len(row)} in {row}")
    name = model_name(row[1])
    cells = tuple(normalize_task_cell(task, cell, name, source) for task, cell in zip(TASK_DIRS, row[2:8], strict=True))
    return name, cells


def normalize(rows, source):
    result = [normalize_model_row(row, source) for row in rows]
    if not result or len({name for name, _ in result}) != len(result):
        raise ValueError(f"{source}: empty model table or duplicate model names")
    return result


def markdown_rows(path):
    rows = []
    found = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line[1:-1].split("|")]
            if tuple(cells[2:8]) == TASKS:
                found = True
                continue
            if found and not all(re.fullmatch(r"[:\- ]+", cell) for cell in cells):
                rows.append(cells)
        elif found:
            break
    return rows


def rst_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("   * - "):
            rows.append([line.removeprefix("   * - ").strip()])
        elif rows and (line.startswith("     - ") or line == "     -"):
            rows[-1].append(line[6:].strip())
    if not rows or tuple(rows[0][2:8]) != TASKS:
        raise ValueError(f"{path}: missing task headers")
    return rows[1:]


def main():
    canonical = normalize(markdown_rows(ROOT / "README.md"), "README.md")
    paths = sorted((ROOT / "i18n").glob("README_*.md"))
    expected = {"de", "es", "fr", "id", "ja", "ko", "pt-BR", "ru", "zh"}
    missing = expected - {path.stem.removeprefix("README_") for path in paths}
    if missing:
        raise ValueError(f"Missing translated READMEs: {sorted(missing)}")
    for path in paths:
        actual = normalize(markdown_rows(path), path)
        if actual != canonical:
            raise ValueError(f"{path}: model names, ordering, task markers or example links differ from README.md")
    path = ROOT / "docs/algo_table.rst"
    if normalize(rst_rows(path), path) != canonical:
        raise ValueError(f"{path}: model names, ordering, task markers or example links differ from README.md")
    requirements = [
        line.strip()
        for line in (ROOT / "requirements/requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    install = (ROOT / "docs/install.rst").read_text(encoding="utf-8")
    documented = re.findall(r"^\* (.+)$", install.split("Optional Dependencies")[0], re.MULTILINE)
    if documented[-len(requirements) :] != requirements:
        raise ValueError("docs/install.rst: dependency list differs from requirements/requirements.txt")
    print(
        f"PASS: {len(canonical)} model rows, {len(paths)} translations, RST model zoo, example links, {len(requirements)} dependencies"
    )


if __name__ == "__main__":
    main()
