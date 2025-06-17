import csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Mapping, Literal

from .file_utils import ensure_directory, resolve_write_mode, unique_path
from .csv_utils import assert_csv_header_matches

# Default CSV schema
FIELDNAMES = [
    "method", "task", "num_simulations", "observation_idx",
    "metric", "value", "timestamp"
]


def save_results(
        results: Mapping[str, float],
        *,
        method: str,
        task: str,
        num_simulations: int,
        observation_idx: int,
        directory: str | Path = "outputs/results",
        filename: str | None = None,
        file_mode: Literal["write", "append"] = "append",
        sep: str = "__",
        encoding: str = "utf-8",
        **metadata: str,
) -> Path:
    """
    Save benchmark results to a CSV file (one row per metric).

    Args:
        results (Mapping[str, float]): Mapping of metric names to their respective values.
        method (str): Inference method name
        task (str): Benchmark task name
        num_simulations (int): Number of simulations.
        observation_idx (int): Index of observation.
        directory (str, optional): Directory to save the file. Defaults to "outputs/results".
        filename (str, optional): CSV filename.
            If None, a unique name within the given directory is generated based on method and task name.
        file_mode ("write" or "append", optional): 'write' to overwrite;
            'append' to append or create if the file does not exist.
            Defaults to "append".
        sep (str, optional): Separator between parts of the auto-generated stem.
        encoding (str, optional): File encoding.
        **metadata: Additional metadata to include (e.g.: random seed).

    Raises:
        ValueError: If `results` is empty.

    Returns:
        Path: The path of the written CSV file.
    """

    # Asserts that results are not empty
    if not results:
        raise ValueError("`results` is empty; nothing to save.")

    # Ensure the save directory (and any missing parents) exists
    directory = Path(directory)
    ensure_directory(directory)

    # Determine the saving path
    if filename:
        stem = Path(filename).stem  # Gets the stem, regardless of whether the filename had an extension or not
        path = directory / f"{stem}.csv"
    else:
        stem = f"{method}{sep}{task}"
        desired_path = directory / f"{stem}.csv"
        path = unique_path(desired_path, sep=sep)


    # Decide mode and header-writing behavior
    mode, write_header = resolve_write_mode(path, file_mode)

    # Build rows for each metric
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows = [
        {
            "method": method,
            "task": task,
            "num_simulations": num_simulations,
            "observation_idx": observation_idx,
            "metric": metric_name,
            "value": metric_value,
            "timestamp": timestamp,
            **metadata,
        }
        for metric_name, metric_value in results.items()
    ]

    # Assert that the headers match when appending to an existing CSV file
    if file_mode == "append" and path.exists():
        assert_csv_header_matches(path, FIELDNAMES)

    # Write or append to the CSV
    with path.open(mode, newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # Confirm the file was saved
    print(f"Saved {len(rows)} metric(s) ➜ {path.resolve()}")

    return path
