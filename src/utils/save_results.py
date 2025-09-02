import csv
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

from .csv_utils import assert_csv_header_matches, resolve_file_mode
from .file_utils import ensure_directory


def save_results(
        results: Mapping[str, float],
        *,
        task: str,
        method: str,
        num_simulations: int,
        observation_idx: int,
        base_directory: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        file_mode: Literal["write", "append"] = "write",
        **metadata: Union[str, int, float, bool],
) -> Path:
    """
    Save benchmark results to a CSV file (one row per metric).

    A folder structure under `base_directory` is created as:
        outputs/
        <task>_<method>/
        [tau_{tau_m}_lambda_{lambda_val}_etc/]  <-- new: all task params as subfolders, order sorted
        sims_{num_simulations}/
        obs_{observation_idx}/

    By default, the file “metrics.csv” in that leaf folder is overwritten on each call.
    To add rows instead, set `file_mode="append"`.
    To write multiple files in the same folder, supply a custom `filename`.

    Args:
        results (Mapping[str, float]): Mapping of metric names to their respective values.
        task (str): Benchmark task name.
        method (str): Inference method name.
        num_simulations (int): Number of simulations.
        observation_idx (int): Index of observation.
        base_directory (Path | str, optional): Root directory for folder structure; defaults to cwd.
        filename (str, optional): Custom .csv filename (stem or with extension). Defaults to "metrics.csv".
        file_mode ("write" or "append", optional):
            - "write" (default): overwrite the file if it exists, or create it otherwise.
            - "append": append rows if the file exists, or create it otherwise.
        **metadata: Additional metadata columns (e.g.: random seed and task parameters).

    Raises:
        ValueError: If `results` is empty.

    Returns:
        Path: The absolute path of the .csv file the benchmark results have been saved to.
    """

    # Assert results is not empty
    if not results:
        raise ValueError("`results` is empty; nothing to save.")

    # Identify task parameters in metadata (anything except known keys)
    # Assume task parameters are all keys in metadata that are not 'random_seed', etc.
    non_task_param_keys = {"random_seed"}
    task_param_items = [(k, v) for k, v in metadata.items() if k not in non_task_param_keys]
    # Sort by key for reproducibility
    task_param_items.sort()
    # Build subfolder string, e.g. "tau_1.0_lambda_0.5"
    if task_param_items:
        task_param_folder = "_".join([f"{k}_{v}" for k, v in task_param_items])
    else:
        task_param_folder = None

    # Build the save path
    base_dir = Path.cwd() if base_directory is None else Path(base_directory)
    stem = Path(filename).stem if filename else "metrics"

    save_path = base_dir / "outputs" / f"{task}_{method}"
    if task_param_folder:
        save_path = save_path / task_param_folder
    save_path = save_path / f"sims_{num_simulations}" / f"obs_{observation_idx}" / f"{stem}.csv"

    ensure_directory(save_path.parent)

    # Build rows for each metric
    rows = [
        {
            "metric": metric_name,
            "value": metric_value,
            "task": task,
            "method": method,
            "num_simulations": num_simulations,
            "observation_idx": observation_idx,
            **metadata,
        }
        for metric_name, metric_value in results.items()
    ]


    # Assert that the headers/ fieldnames match when appending to an existing CSV file
    base_fieldnames = ["metric", "value", "task", "method", "num_simulations", "observation_idx"]
    metadata_fieldnames = sorted(metadata.keys())  # normalize metadata fieldnames order

    fieldnames = base_fieldnames + metadata_fieldnames

    if file_mode == "append" and save_path.exists():
        assert_csv_header_matches(save_path, fieldnames)


    # Write or append to the save path
    mode, write_header = resolve_file_mode(save_path, file_mode)  # Decide mode and header-writing behavior

    with save_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


    # Confirm the file was saved
    print(f"Saved {len(rows)} metric(s) ➜ {save_path}")

    return save_path
