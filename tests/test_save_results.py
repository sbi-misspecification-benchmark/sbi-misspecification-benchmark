import pandas as pd
import pytest

import src.utils.save_results as save_results


def test_save_results_empty_raises(tmp_path):
    with pytest.raises(ValueError) as exc:
        save_results.save_results({},
                                  task="t",
                                  method="m",
                                  num_simulations=10,
                                  observation_idx=1,
                                  base_directory=tmp_path)
    assert "`results` is empty" in str(exc.value)


def test_save_results_write_creates_file(tmp_path):
    results = {"metric1": 1.23, "metric2": 4.56}
    task = "taskA"
    method = "methodB"
    num_sim = 5
    obs_idx = 2
    base = tmp_path / "base_dir"
    # Call save_results
    save_path = save_results.save_results(
        results,
        task=task,
        method=method,
        num_simulations=num_sim,
        observation_idx=obs_idx,
        base_directory=base,
        filename=None,  # default "metrics.csv"
        file_mode="write",
    )
    # Expected path
    expected = (base
                / "outputs"
                / f"{task}_{method}"
                / f"sims_{num_sim}"
                / f"obs_{obs_idx}"
                / "metrics.csv")
    assert save_path == expected
    assert expected.exists()
    # Read CSV and verify contents
    df = pd.read_csv(expected)
    # Two rows, one per metric
    assert len(df) == 2
    # Columns: base_fieldnames
    expected_cols = ["metric", "value", "task", "method", "num_simulations", "observation_idx"]
    assert list(df.columns) == expected_cols
    # Check values and constant columns
    for _, row in df.iterrows():
        assert row["task"] == task
        assert row["method"] == method
        assert row["num_simulations"] == num_sim
        assert row["observation_idx"] == obs_idx
        assert row["metric"] in results
        assert pytest.approx(results[row["metric"]]) == row["value"]


def test_save_results_custom_filename(tmp_path):
    results = {"m": 0.1}
    task = "T"
    method = "M"
    num_sim = 1
    obs_idx = 0
    base = tmp_path
    # Without extension in filename
    save_path1 = save_results.save_results(
        results,
        task=task,
        method=method,
        num_simulations=num_sim,
        observation_idx=obs_idx,
        base_directory=base,
        filename="customname",
        file_mode="write",
    )
    expected1 = (base
                 / "outputs"
                 / f"{task}_{method}"
                 / f"sims_{num_sim}"
                 / f"obs_{obs_idx}"
                 / "customname.csv")
    assert save_path1 == expected1
    assert expected1.exists()

    # With extension in filename
    save_path2 = save_results.save_results(
        results,
        task=task,
        method=method,
        num_simulations=num_sim,
        observation_idx=obs_idx,
        base_directory=base,
        filename="other.csv",
        file_mode="write",
    )
    expected2 = (base
                 / "outputs"
                 / f"{task}_{method}"
                 / f"sims_{num_sim}"
                 / f"obs_{obs_idx}"
                 / "other.csv")
    assert save_path2 == expected2
    assert expected2.exists()


def test_save_results_with_metadata(tmp_path):
    results = {"m1": 2.0}
    task = "tt"
    method = "mm"
    num_sim = 3
    obs_idx = 7
    metadata = {"seed": 42, "flag": True}
    base = tmp_path
    save_path = save_results.save_results(
        results,
        task=task,
        method=method,
        num_simulations=num_sim,
        observation_idx=obs_idx,
        base_directory=base,
        filename=None,
        file_mode="write",
        **metadata
    )
    df = pd.read_csv(save_path)
    # Columns should be base_fieldnames + sorted metadata keys
    base_fields = ["metric", "value", "task", "method", "num_simulations", "observation_idx"]
    meta_fields = sorted(metadata.keys())
    assert list(df.columns) == base_fields + meta_fields
    # Check metadata values
    row = df.iloc[0]
    for k, v in metadata.items():
        assert row[k] == v


def test_save_results_append_behavior(tmp_path):
    results1 = {"a": 1.0}
    results2 = {"b": 2.0}
    task = "X"
    method = "Y"
    num_sim = 2
    obs_idx = 3
    base = tmp_path
    # First call: write mode
    save_path = save_results.save_results(
        results1,
        task=task,
        method=method,
        num_simulations=num_sim,
        observation_idx=obs_idx,
        base_directory=base,
        file_mode="write",
    )
    # Second call: append mode, same structure, no metadata
    save_path2 = save_results.save_results(
        results2,
        task=task,
        method=method,
        num_simulations=num_sim,
        observation_idx=obs_idx,
        base_directory=base,
        file_mode="append",
    )
    # Paths should match
    assert save_path2 == save_path
    df = pd.read_csv(save_path)
    # Now two rows total
    assert len(df) == 2
    # Metrics present
    mets = set(df["metric"])
    assert mets == {"a", "b"}


def test_save_results_append_new_file(tmp_path):
    # Append mode but the file doesn't exist: should behave like 'write'
    results = {"z": 9.9}
    task = "T2"
    method = "M2"
    num_sim = 4
    obs_idx = 5
    base = tmp_path
    save_path = save_results.save_results(
        results,
        task=task,
        method=method,
        num_simulations=num_sim,
        observation_idx=obs_idx,
        base_directory=base,
        file_mode="append",
    )
    assert save_path.exists()
    df = pd.read_csv(save_path)
    assert len(df) == 1
    assert df.iloc[0]["metric"] == "z"


def test_save_results_append_header_mismatch(tmp_path):
    # First, write with metadata {"a":1}
    results = {"x": 0.5}
    task = "T3"
    method = "M3"
    num_sim = 1
    obs_idx = 1
    base = tmp_path
    save_results.save_results(
        results,
        task=task,
        method=method,
        num_simulations=num_sim,
        observation_idx=obs_idx,
        base_directory=base,
        file_mode="write",
        a=1
    )
    # Now attempt append with a different metadata key {"b":2}
    with pytest.raises(ValueError) as exc:
        save_results.save_results(
            {"y": 1.5},
            task=task,
            method=method,
            num_simulations=num_sim,
            observation_idx=obs_idx,
            base_directory=base,
            file_mode="append",
            b=2
        )
    msg = str(exc.value)
    assert "CSV header mismatch" in msg
