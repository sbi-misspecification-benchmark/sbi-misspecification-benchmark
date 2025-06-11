import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Optional


def save_results(
    results: dict[str, float],
    method: str,
    task: str,
    num_simulations: int,
    observation_idx: int,
    seed: int,
    save_dir: str = "results",
    filename: Optional[str] = None,
    **kwargs: dict[str, Any],
) -> None:
    """
    Save benchmark results as a CSV file (one row per metric).

    If a filename is provided, results are appended if the file exists, or written to a new file otherwise.
    If no filename is provided, a unique filename is generated based on benchmark parameters.

    Args:
        results (dict): Benchmark results as {metric_name (str): value (float or int)}
            (e.g.: {"C2ST": 0.84, "runtime_sec": 123.5}).
        method (str): Inference algorithm name (e.g.: "REJ_ABC", "nNLE").
        task (str): Benchmark task name (e.g.: "two_moons", "gaussian_linear").
        num_simulations (int): Number of simulations.
        observation_idx (int): Index of observation.
        seed (int): Random seed used.
        save_dir (str, optional): Directory to save the file (default: "results").
        filename (str, optional): CSV filename (default: None).
            If None, a unique CSV filename is generated.

        **kwargs: Additional metadata to include.
    """

    # Create the save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Determine filepath and write mode
    if filename is None:
        # Auto-Generating a unique filename based on the used benchmark parameters
        base = f"{task}__{method}__seed{seed}__obs{observation_idx}"
        filepath = save_path / f"{base}.csv"

        # Add run suffix if needed
        if filepath.exists():
            run_id = 1
            while True:
                new_filename = f"{base}__{run_id}.csv"
                filepath = save_path / new_filename
                if not filepath.exists():
                    break
                run_id += 1

        write_mode = "w"
        write_header = True

    else:
        # Use given filename, and append if it already exists
        filepath = save_path / filename
        file_exists = filepath.exists()
        write_mode = "a" if file_exists else "w"
        write_header = not file_exists

    # Generate ISO-formatted timestamp (UTC)
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat()

    # Create a row for each metric
    rows = []
    for metric_name, value in results.items():
        row = {
            "method": method,
            "task": task,
            "seed": seed,
            "num_simulations": num_simulations,
            "observation_idx": observation_idx,
            "metric": metric_name,
            "value": value,
            "timestamp": timestamp,
            **kwargs
        }
        rows.append(row)

    # Validate header if appending to an existing file
    if write_mode == "a":
        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader)
        current_header = list(rows[0].keys())
        if set(existing_header) != set(current_header):
            raise ValueError(
                f"Header mismatch when appending to '{filepath}':\n"
                f"Expected columns: {existing_header}\n"
                f"Given: {current_header}"
            )

    # Append or create the CSV file
    with open(filepath, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # Confirm the file was saved
    print(f"Saved {len(rows)} metric(s) to {filepath.resolve()}")
