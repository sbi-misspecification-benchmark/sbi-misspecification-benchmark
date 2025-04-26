import json
import csv
import pytest
from datetime import datetime
from benchmark.save_results import save_results


def test_json_output(tmp_path):
    """Tests if the JSON output is correct"""
    results = {"test_metric_a1": 0.5, "test_metric_b1": 0.7}
    method = "test_method1"
    task = "test_task1"
    seed = 0
    save_dir = tmp_path / "results"

    # Write JSON
    save_results(results, method, task, seed, save_dir=save_dir, file_format="json")
    file_path = save_dir / f"{method}__{task}__seed{seed}.json"

    # File should exist
    assert file_path.exists()

    # Load and verify contents
    with open(file_path, "r") as f:
        data = json.load(f)

    assert data["method"] == method
    assert data["task"] == task
    assert data["seed"] == seed
    datetime.fromisoformat(data["timestamp"])  # Make sure timestamp is a valid ISO string
    assert data["metrics"] == results


def test_csv_output(tmp_path):
    """Tests if the CSV output is correct"""
    metrics = {"test_metric_a2": 1.0, "test_metric_b2": 0.5}
    method = "test_method2"
    task = "test_task2"
    seed = 1
    save_dir = tmp_path / "results"

    # Write CSV
    save_results(metrics, method, task, seed, save_dir=save_dir, file_format="csv")
    file_path = save_dir / f"{method}__{task}__seed{seed}.csv"

    # File should exist
    assert file_path.exists()

    # Read back with DictReader
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    assert row["method"] == method
    assert row["task"] == task
    assert int(row["seed"]) == seed
    datetime.fromisoformat(row["timestamp"])  # Make sure timestamp is a valid ISO string
    assert abs(float(row["test_metric_a2"]) - metrics["test_metric_a2"]) < 1e-8
    assert abs(float(row["test_metric_b2"]) - metrics["test_metric_b2"]) < 1e-8


def test_run_id_generation(tmp_path):
    """Tests if the increasing run parameter run_id works for files with the same filename"""
    metrics = {"test_metric_a3": 22.5, "test_metric_b3": 12.1}
    method = "test_method3"
    task = "test_task3"
    seed = 2
    save_dir = tmp_path / "results"

    # First call writes base filename
    save_results(metrics, method, task, seed, save_dir=save_dir, file_format="json")
    base = save_dir / f"{method}__{task}__seed{seed}.json"
    assert base.exists()

    # Second call causes __run1 addition to the filename
    save_results(metrics, method, task, seed, save_dir=save_dir, file_format="json")
    run1 = save_dir / f"{method}__{task}__seed{seed}__run1.json"
    assert run1.exists()

    # Third call causes __run2 addition to the filename
    save_results(metrics, method, task, seed, save_dir=save_dir, file_format="json")
    run2 = save_dir / f"{method}__{task}__seed{seed}__run2.json"
    assert run1.exists()


def test_input_validation(tmp_path):
    """Tests if the wrong inputs raise the correct Error"""
    # method must be str
    with pytest.raises(TypeError):
        save_results({}, 123, "t", 0, save_dir=tmp_path)

    # task must be str
    with pytest.raises(TypeError):
        save_results({}, "m", 456, 0, save_dir=tmp_path)

    # seed must be int
    with pytest.raises(TypeError):
        save_results({}, "m", "t", "zero", save_dir=tmp_path)

    # file_format must be json or csv
    with pytest.raises(ValueError):
        save_results({}, "m", "t", 0, save_dir=tmp_path, file_format="xml")
