import pandas as pd
import pytest
from pathlib import Path

from src.utils.consolidate_metrics import consolidate_metrics


def create_dummy_csv(path: Path, task: str, method: str, num_sim: int, obs_idx: int):
    """
    Creates a dummy metrics.csv file with two metric rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {
            "metric": "C2ST",
            "value": 0.5,
            "task": task,
            "method": method,
            "num_simulations": num_sim,
            "observation_idx": obs_idx,
        },
        {
            "metric": "PPC",
            "value": 0.25,
            "task": task,
            "method": method,
            "num_simulations": num_sim,
            "observation_idx": obs_idx,
        },
    ])
    df.to_csv(path, index=False)


def test_consolidate_multiple_results(tmp_path):
    # Create a temporary base directory with two runs of the same Task/Method but different Sims/Obs folder
    base = tmp_path / "outputs" / "TaskA_MethodA"

    # run1
    p1 = base / "sims_100" / "metrics.csv"
    create_dummy_csv(p1, "TaskA", "MethodA", 100, 0)
    # run2
    p2 = base / "sims_200" / "metrics.csv"
    create_dummy_csv(p2, "TaskA", "MethodA", 200, 1)

    # Consolidate the results under the given input directory and save it at the given output directory
    output_file = tmp_path / "combined.csv"
    df = consolidate_metrics(input_dir=base, output_file=output_file)

    # Assert the new row length
    assert len(df) == 4

    # Assert the file exists at the given output path
    assert output_file.exists()

    # Assert the distinct columns (num_simulations, observation_idx) are as expected
    assert set(df["num_simulations"]) == {100, 200}
    assert set(df["observation_idx"]) == {0, 1}


def test_consolidate_no_files(tmp_path):
    # Create an empty directory
    empty = tmp_path
    output_file = tmp_path / "metrics_all.csv"

    # Assert that a FileNotFoundError is raised when no metrics.csv files exist
    with pytest.raises(FileNotFoundError) as exc:
        consolidate_metrics(input_dir=empty, output_file=output_file)
    assert "No metrics.csv under" in str(exc.value)


def test_consolidate_unreadable_files(tmp_path):
    base = tmp_path / "outputs" / "Task_Method"
    csv_path = base / "sims_1" / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Create an empty file
    csv_path.write_text("")

    output_file = tmp_path / "out.csv"
    # Since read_csv_files will skip the bad file, frames will be empty and consolidate should raise
    with pytest.raises(FileNotFoundError) as exc:
        consolidate_metrics(base, output_file)
    assert "Failed to read any CSVs" in str(exc.value)


def test_consolidate_missing_required_column(tmp_path):
    # Prepare a Task_Method folder with a metrics.csv missing a required column
    base = tmp_path / "outputs" / "Task_Method"
    csv_path = base / "sims_1" / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a DataFrame missing the 'method' column
    df_missing = pd.DataFrame([{
        "metric": "M1",
        "value": 0.5,
        "task": "Task",
        # "method" column omitted
        "num_simulations": 1,
        "observation_idx": 0
    }])
    df_missing.to_csv(csv_path, index=False)

    # Running consolidate() should raise a ValueError about the missing column
    output_file = tmp_path / "out.csv"
    with pytest.raises(ValueError) as exc:
        consolidate_metrics(base, output_file)
    assert "Missing required columns" in str(exc.value)


def test_consolidate_missing_value_in_required_column(tmp_path):
    # Prepare a Task_Method folder with a metrics.csv containing a missing required value
    base = tmp_path / "outputs" / "Task_Method"
    csv_path = base / "sims_3" / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a DataFrame where the 'value' column has a missing entry
    df_null = pd.DataFrame([{
        "metric": "M1",
        "value": None,             # missing required value
        "task": "Task",
        "method": "Method",
        "num_simulations": 3,
        "observation_idx": 0
    }])
    df_null.to_csv(csv_path, index=False)

    # Running consolidate() should raise a ValueError about missing values
    output_file = tmp_path / "out.csv"
    with pytest.raises(ValueError) as exc:
        consolidate_metrics(base, output_file)

    assert "Required columns contain missing values" in str(exc.value)


def test_consolidate_required_columns_reordered(tmp_path):
    # Prepare a Task_Method folder with a metrics.csv that has a different required_columns order
    base = tmp_path / "outputs" / "Task_Method"
    bad_csv = base / "sims_10" / "metrics.csv"
    bad_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build a DataFrame with the wrong column order
    df_bad = pd.DataFrame([{
        "task": "Task",
        "metric": "C2ST",
        "value": 0.1,
        "method": "Method",
        "num_simulations": 10,
        "observation_idx": 0,
    }])
    df_bad.to_csv(bad_csv, index=False)

    # Run consolidate()
    output_file = tmp_path / "dummy.csv"
    result_df = consolidate_metrics(base, output_file)

    # Check that output file was created
    assert output_file.exists()

    # Ensure required columns exist and are in the canonical order
    required = ["metric", "value", "task", "method", "num_simulations", "observation_idx"]
    assert list(result_df.columns)[:len(required)] == required

    # Ensure the values for those columns match the original data
    pd.testing.assert_frame_equal(
        result_df[required].reset_index(drop=True),
        df_bad[required].reset_index(drop=True)
    )


def test_consolidate_extra_columns_reordered(tmp_path):
    base = tmp_path / "outputs" / "Task_Method"
    csv_path = base / "sims_5" / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a DataFrame with correct base fields, but two extra metadata columns in unsorted order
    df = pd.DataFrame([{
        "metric": "PPC",
        "value": 0.2,
        "task": "Task",
        "method": "Method",
        "num_simulations": 5,
        "observation_idx": 2,
        "z_meta": 123,  # 'z_meta' comes after 'a_meta' alphabetically, but we'll reverse them
        "a_meta": "foo",
    }])
    df.to_csv(csv_path, index=False)

    # Run consolidate() (should succeed, sorting extra columns automatically)
    output_file = tmp_path / "dummy.csv"
    result_df = consolidate_metrics(base, output_file)

    # Check that output file was created
    assert output_file.exists()

    # Required columns should come first...
    required = ["metric", "value", "task", "method", "num_simulations", "observation_idx"]
    assert list(result_df.columns)[:len(required)] == required

    # ...and extra columns should be sorted alphabetically
    extra = list(result_df.columns[len(required):])
    assert extra == sorted(extra) == ["a_meta", "z_meta"]

    # Underlying values should be preserved
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True),
        pd.DataFrame([{**{c: df.at[0, c] for c in required}, "a_meta": "foo", "z_meta": 123}])
    )
