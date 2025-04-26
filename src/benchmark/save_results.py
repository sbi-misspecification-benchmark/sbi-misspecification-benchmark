import json
import csv
from pathlib import Path
from datetime import datetime


def save_results(metrics, method, task, seed, save_dir="results", file_format="json", **kwargs):
    """
    Save benchmark results as JSON or CSV.

    Args:
        metrics (dict): Benchmark metrics as {metric_name (str): score (float or int)},
            e.g., {"C2ST": 0.84, "runtime_sec": 123.5}.
        method (str): Inference algorithm name, e.g., "REJ_ABC" or "nNLE".
        task (str): Benchmark task name, e.g., "two_moons" or "gaussian_linear".
        seed (int): Random seed used.
        save_dir (str or pathlib.Path, optional): Directory in which to save results.
            Defaults to "results".
        file_format (str, optional): Output format, either "json" or "csv".
            Defaults to "json".
        **kwargs: Additional metadata to include in the saved file.
    """

    # --- Input type validation ---
    if not isinstance(method, str):
        raise TypeError(f"Expected 'method' to be str, got {type(method).__name__}")
    if not isinstance(task, str):
        raise TypeError(f"Expected 'task' to be str, got {type(task).__name__}")
    if not isinstance(seed, int):
        raise TypeError(f"Expected 'seed' to be int, got {type(seed).__name__}")
    if not isinstance(save_dir, (str, Path)):
        raise TypeError(f"Expected 'save_dir' to be a str or Path, got {type(save_dir).__name__}")
    if not isinstance(file_format, str):
        raise TypeError(f"Expected 'file_format' to be str, got {type(file_format).__name__}")

    file_format = file_format.lower()  # Allows user-input variations (e.g. "Json", "CSV", ...)
    if file_format not in {"json", "csv"}:
        raise ValueError(f"Unsupported file format: {file_format}. Use 'json' or 'csv'.")


    # --- Prepare save location and filename ---
    # Create save directory (if it doesn't exist yet)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create initial filename and filepath
    base_filename = f"{method}__{task}__seed{seed}"
    filename = f"{base_filename}.{file_format}"
    filepath = save_dir / filename

    # Add run_id if filepath already exists (to allow multiple benchmark versions of the same method, task and seed)
    if filepath.exists():
        run_id = 1
        while True:
            filename = f"{base_filename}__run{run_id}.{file_format}"
            filepath = save_dir / filename
            if not filepath.exists():
                break    # Found unique filepath
            run_id += 1  # Otherwise increment run_id and try again


    # --- Write results to file ---
    if file_format == "json":
        # Prepare nested output for JSON
        output = {
            "method": method,
            "task": task,
            "seed": seed,
            "timestamp": datetime.utcnow().replace(microsecond=0).isoformat(),  # Universal timestamp
            **kwargs,  # Add any extra user-provided metadata
            "metrics": metrics
        }
        # Write JSON file
        with open(filepath, "w") as f:
            json.dump(output, f, indent=4)

    elif file_format == "csv":
        # Prepare flat row for CSV (metrics merged into top level)
        row = {
            "method": method,
            "task": task,
            "seed": seed,
            "timestamp": datetime.utcnow().replace(microsecond=0).isoformat(),  # Universal timestamp
            **kwargs,
            **metrics
        }
        # Write CSV file
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

    # Confirm result saving in console
    print(f"Results saved to {filepath.resolve()}")
