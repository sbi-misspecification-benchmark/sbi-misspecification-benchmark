import json
import csv
from pathlib import Path
from datetime import datetime


def save_results(results, method, task, seed, format="json", **kwargs):
    """
    Saves benchmark results to a structured file (JSON or CSV) in a 'results/' folder.

    Args:
        results (dict): TODO
        method (str): TODO
        task (str): TODO
        seed (int or str): TODO
        format (str): TODO
        **kwargs: TODO
    """
    # Ensure the results folder exists
    Path("results").mkdir(parents=True, exist_ok=True)

    # Add timestamp once and reuse
    timestamp = datetime.utcnow().isoformat()

    # Combine all data into a flat dictionary for CSV, or nested for JSON
    result_data = {
        "method": method,
        "task": task,
        "seed": seed,
        "timestamp": timestamp,
        **kwargs
    }

    suffix = ".json" if format == "json" else ".csv"
    filename = f"{method}_{task}_seed{seed}{suffix}"
    filepath = Path("results") / filename

    if format == "json":
        # For JSON, nest the metrics
        result_data["metrics"] = results
        with open(filepath, "w") as file:
            json.dump(result_data, file, indent=4)

    elif format == "csv":
        # For CSV, flatten the metrics into the top level
        csv_row = {**result_data, **results}
        with open(filepath, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_row.keys())
            writer.writeheader()
            writer.writerow(csv_row)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

    print(f"Results saved to {filepath}")
