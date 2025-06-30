import pandas as pd
import pytest
from pathlib import Path

from src.utils.consolidate_metrics import consolidate


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
    p1 = base / "sims_100" / "obs_0" / "metrics.csv"
    create_dummy_csv(p1, "TaskA", "MethodA", 100, 0)
    # run2
    p2 = base / "sims_200" / "obs_1" / "metrics.csv"
    create_dummy_csv(p2, "TaskA", "MethodA", 200, 1)

    # Consolidate the results under the given input directory and save it at the given output directory
    output_file = tmp_path / "combined.csv"
    df = consolidate(input_dir=base, output_file=output_file)


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
        consolidate(input_dir=empty, output_file=output_file)
    assert "No metrics.csv under" in str(exc.value)


def test_consolidate_fieldname_mismatch(tmp_path):
    # Prepare a Task_Method folder
    base = tmp_path / "outputs" / "Task_Method"
    # Create one run with sims_10/obs_0/metrics.csv
    bad_csv = base / "sims_10" / "obs_0" / "metrics.csv"
    bad_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build a DataFrame with the wrong column order:
    df_bad = pd.DataFrame([
        {
            "task": "Task",
            "metric": "C2ST",
            "value": 0.1,
            "method": "Method",
            "num_simulations": 10,
            "observation_idx": 0,
        }
    ])
    df_bad.to_csv(bad_csv, index=False)

    # Now calling consolidate() should raise a ValueError about wrong structure
    with pytest.raises(ValueError) as exc:
        consolidate(base, tmp_path/"dummy.csv")
    assert "Wrong CSV structure" in str(exc.value)


def test_consolidate_metadata_not_sorted(tmp_path):
    base = tmp_path / "outputs" / "Task_Method"
    csv_path = base / "sims_5" / "obs_2" / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a DataFrame with correct base fields, but two extra metadata columns in unsorted order
    df = pd.DataFrame([{
        "metric": "PPC",
        "value": 0.2,
        "task": "Task",
        "method": "Method",
        "num_simulations": 5,
        "observation_idx": 2,
        "z_meta": 123,      # 'z_meta' comes after 'a_meta' alphabetically, but we'll reverse them
        "a_meta": "foo",
    }])
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError) as exc:
        consolidate(base, tmp_path/"dummy.csv")
    assert "Unexpected or mis-ordered metadata columns" in str(exc.value)
