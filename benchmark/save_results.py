import json
import csv
from pathlib import Path
from datetime import datetime


def save_results(
    metrics,
    method,
    task,
    seed,
    save_dir="results",
    file_format="json",
    **kwargs,
):
    """
    Save benchmark results as JSON or CSV.

    Args:
        metrics (dict): Benchmark results as {metric_name (str): score (float or int)}
                        (e.g.: {"C2ST": 0.84, "runtime_sec": 123.5}, ...).
        method (str): Inference algorithm name (e.g.: "REJ_ABC", "nNLE", ...).
        task (str): Benchmark task name (e.g.: "two_moons", "gaussian_linear", ...).
        seed (int): Random seed used.
        save_dir (str, optional): Output directory (default "results").
        file_format (str, optional): "json" or "csv" (default "json").
        **kwargs: Additional metadata to include.
    """

    # --- Input type validation ---
    if not isinstance(method, str):
        raise TypeError(f"Expected 'method' to be str, got {type(method).__name__}")
    if not isinstance(task, str):
        raise TypeError(f"Expected 'task' to be str, got {type(task).__name__}")
    if not isinstance(seed, int):
        raise TypeError(f"Expected 'seed' to be int, got {type(seed).__name__}")
    if not isinstance(save_dir, (str, Path)):
        raise TypeError(
            f"Expected 'save_dir' to be a str or Path, got {type(save_dir).__name__}"
        )
    if not isinstance(file_format, str):
        raise TypeError(
            f"Expected 'file_format' to be str, got {type(file_format).__name__}"
        )

    file_format = file_format.lower()  # allow "JSON", "Csv", etc.
    if file_format not in {"json", "csv"}:
        raise ValueError(f"Unsupported file format: {file_format}. Use 'json' or 'csv'.")

    # --- Prepare save location and filename ---
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base = f"{method}__{task}__seed{seed}"
    filename = f"{base}.{file_format}"
    filepath = save_dir / filename

    # if this exact filename exists, append __runN
    if filepath.exists():
        run_id = 1
        while True:
            filename = f"{base}__run{run_id}.{file_format}"
            filepath = save_dir / filename
            if not filepath.exists():
                break
            run_id += 1

    # --- Write out ---
    if file_format == "json":
        output = {
            "method": method,
            "task": task,
            "seed": seed,
            "timestamp": datetime.utcnow().replace(microsecond=0).isoformat(),
            **kwargs,
            "metrics": metrics,
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=4)

    else:  # csv
        row = {
            "method": method,
            "task": task,
            "seed": seed,
            "timestamp": datetime.utcnow().replace(microsecond=0).isoformat(),
            **kwargs,
            **metrics,
        }
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

    print(f"Results saved to {filepath.resolve()}")
