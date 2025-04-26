import json
import csv
import pytest
from benchmark.save_results import save_results


def test_json_output(tmp_path):
    metrics = {"test_metric_a": 0.5, "test_metric_b": 10.0}
    method = "test_method"
    task = "test_task"
    seed = 0
    save_dir = tmp_path / "results"

    save_results(metrics, method, task, seed, save_dir=save_dir, file_format="json")
    file_path = save_dir / f"{method}__{task}__seed{seed}.json"

    # Check if the file is created
    assert file_path.exists()

    # Check if the contents are correctly saved
    data = json.loads(file_path.read_text())
    assert data["method"] == method
    assert data["task"] == task
    assert data["seed"] == seed
    assert "timestamp" in data
    assert data["metrics"] == metrics


def test_csv_output(tmp_path):
    metrics = {"test_metric_a2": 1.0, "test_metric_b2": 0.5}
    method = "test_method2"
    task = "test_task2"
    seed = 1
    save_dir = tmp_path / "results"

    save_results(metrics, method, task, seed, save_dir=save_dir, file_format="csv")
    file_path = save_dir / f"{method}__{task}__seed{seed}.csv"

    # Check if the file is created
    assert file_path.exists()

    # Read back with DictReader
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    # Check if the contents are correctly saved
    assert row["method"] == method
    assert row["task"] == task
    assert int(row["seed"]) == seed
    assert abs(float(row["test_metric_a2"]) - metrics["test_metric_a2"]) < 1e-8
    assert abs(float(row["test_metric_b2"]) - metrics["test_metric_b2"]) < 1e-8


def test_run_id_generation(tmp_path):
    metrics = {"test_metric_a3": 2.0, "test_metric_b3": 5.5}
    method = "test_method3"
    task = "test_task3"
    seed = 0
    save_dir = tmp_path / "results"

    # First save
    save_results(metrics, method, task, seed, save_dir=save_dir)
    # Second save should create a __run1 file
    save_results(metrics, method, task, seed, save_dir=save_dir)
    # Third save should create a __run2 file (increasing the run_id variable)
    save_results(metrics, method, task, seed, save_dir=save_dir)

    base = save_dir / f"{method}__{task}__seed{seed}.json"
    run1 = save_dir / f"{method}__{task}__seed{seed}__run1.json"
    run2 = save_dir / f"{method}__{task}__seed{seed}__run2.json"

    assert base.exists()
    assert run1.exists()
    assert run2.exists()


def test_input_validation(tmp_path):
    metrics = {"test_metric4": 1.0}
    # Wrong types
    with pytest.raises(TypeError):
        save_results(metrics, 123, "task", 0, save_dir=tmp_path)
    with pytest.raises(TypeError):
        save_results(metrics, "method", 456, 0, save_dir=tmp_path)
    with pytest.raises(TypeError):
        save_results(metrics, "method", "task", "0", save_dir=tmp_path)
    with pytest.raises(TypeError):
        save_results(metrics, "method", "task", "0", save_dir=0)
    with pytest.raises(TypeError):
        save_results(metrics, "method", "task", 0, save_dir=tmp_path, file_format=0)
    # Invalid format
    with pytest.raises(ValueError):
        save_results(metrics, "method", "task", 0, save_dir=tmp_path, file_format="xml")
