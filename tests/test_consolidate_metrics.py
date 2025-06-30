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


def test_consolidate_multiple_runs(tmp_path):
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



def test_consolidate_no_files_raises(tmp_path):
    # Create an empty directory
    empty = tmp_path
    output_file = tmp_path / "metrics_all.csv"

    # Assert that a FileNotFoundError is raised when no metrics.csv files exist
    with pytest.raises(FileNotFoundError) as exc:
        consolidate(input_dir=empty, output_file=output_file)
    assert "No metrics.csv under" in str(exc.value)
