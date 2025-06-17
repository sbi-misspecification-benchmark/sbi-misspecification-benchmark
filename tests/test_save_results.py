import csv
import pytest
from pathlib import Path

from src.utils.save_results import save_results


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_auto_generated_filename(tmp_path):
    """
    When filename is None, an auto-generated name based on method and task should be used.
    """
    results = {"m1": 1.0}
    path = save_results(
        results,
        method="method",
        task="task",
        num_simulations=1,
        observation_idx=0,
        save_directory=tmp_path,
        filename=None
    )
    assert path.parent == tmp_path
    assert path.suffix == ".csv"
    assert path.name.startswith("method__task")  # Name must start with 'method__task'


def test_explicit_filename_without_collision(tmp_path):
    """
    When a filename is provided, it should be used with .csv appended. No suffixing.
    """
    results = {"a": 2.0}
    path = save_results(
        results,
        method="X",
        task="Y",
        num_simulations=1,
        observation_idx=1,
        save_directory=tmp_path,
        filename="report",
        file_mode="append"
    )
    assert path.name == "report.csv"

    # Second call in 'append' mode should not change filename
    path2 = save_results(
        {"b": 3.0},
        method="X",
        task="Y",
        num_simulations=1,
        observation_idx=2,
        save_directory=tmp_path,
        filename="report",
        file_mode="append"
    )
    assert path2 == path


def test_append_mode_adds_rows(tmp_path):
    """
    In 'append' mode, calling save_results twice should accumulate rows in the same file.
    """
    filename = "accumulate"
    path = save_results(
        {"m1": 1},
        method="M",
        task="T",
        num_simulations=1,
        observation_idx=0,
        save_directory=tmp_path,
        filename=filename,
        file_mode="append"
    )
    save_results(
        {"m2": 2},
        method="M",
        task="T",
        num_simulations=1,
        observation_idx=1,
        save_directory=tmp_path,
        filename=filename,
        file_mode="append"
    )
    rows = read_csv(path)
    metrics = {r['metric'] for r in rows}
    assert metrics == {"m1", "m2"}
    assert len(rows) == 2


def test_overwrite_mode_replaces_rows(tmp_path):
    """
    In 'write' (overwrite) mode, the file should be truncated before writing new rows.
    """
    filename = "overwrite"
    path = save_results(
        {"a": 10},
        method="M",
        task="T",
        num_simulations=1,
        observation_idx=0,
        save_directory=tmp_path,
        filename=filename,
        file_mode="append"
    )
    save_results(
        {"b": 20},
        method="M",
        task="T",
        num_simulations=1,
        observation_idx=0,
        save_directory=tmp_path,
        filename=filename,
        file_mode="write"
    )
    rows = read_csv(path)
    assert len(rows) == 1
    assert rows[0]['metric'] == 'b'


def test_empty_results_raises(tmp_path):
    """
    Passing an empty mapping should raise ValueError.
    """
    with pytest.raises(ValueError):
        save_results(
            {},
            method="M",
            task="T",
            num_simulations=1,
            observation_idx=0,
            save_directory=tmp_path
        )


def test_invalid_file_mode_raises(tmp_path):
    """
    Passing an unsupported file_mode should raise ValueError.
    """
    with pytest.raises(ValueError):
        save_results(
            {"m": 1},
            method="M",
            task="T",
            num_simulations=1,
            observation_idx=0,
            save_directory=tmp_path,
            file_mode="invalid"
        )
