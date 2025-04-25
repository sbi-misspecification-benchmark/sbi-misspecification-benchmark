import json
import csv
from pathlib import Path
from datetime import datetime


def save_results(results, method, task, seed, file_format="json", **kwargs):
    """Save benchmark results as JSON or CSV in the 'results/' folder."""

    # Normalize file format input (accepts "JSON", "Json", etc.)
    file_format = file_format.lower()

    # Create the results folder if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    # Prepare file path and name
    timestamp = datetime.utcnow().isoformat()
    filename = f"{method}_{task}_seed{seed}.{file_format}"
    filepath = Path("results") / filename

    if file_format == "json":
        # Prepare nested output for JSON
        output = {
            "method": method,
            "task": task,
            "seed": seed,
            "timestamp": timestamp,
            **kwargs,           # Add any extra user-provided metadata
            "metrics": results  # Save actual benchmark results under "metrics"
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
            "timestamp": timestamp,
            **kwargs,
            **results
        }
        # Write CSV file
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

    else:
        # Handle invalid format input
        raise ValueError(f"Unsupported format: {file_format}. Use 'json' or 'csv'.")

    # Confirm result saving in console
    print(f"Results saved to {filepath}")
