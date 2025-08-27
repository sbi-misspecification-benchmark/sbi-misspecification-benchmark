"""
Consolidate all 'metrics.csv' files (of the same task and method) across simulations and parameter sweeps into a single DataFrame.

This will:
1. Recursively gather all 'metrics.csv' files under the given `input_dir`.
2. Read and concatenate them into a single DataFrame.
3. Dynamically detect all task parameter columns and ensure they are included in the output, sorted.
4. Write the resulting DataFrame to 'output_file' as a .csv file, then return it.

Usage:
    python consolidate_metrics.py --input_dir <base_directory> --output_file <metrics_all.csv>
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
from src.utils.csv_utils import read_csv_files

def parse_args():
    p = argparse.ArgumentParser(
        description="Consolidate all 'metrics.csv' files (of the same task and method) across simulations and parameter sweeps into a single DataFrame."
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Base directory containing metrics.csv files to consolidate"
    )
    p.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="File path where to write the consolidated .csv file (e.g. metrics_all.csv)"
    )
    return p.parse_args()

def find_all_metrics_csvs(input_dir: Path):
    # Recursively find all 'metrics.csv' files under input_dir
    return list(input_dir.rglob("metrics.csv"))

def consolidate_metrics(input_dir: Path, output_file: Path) -> pd.DataFrame:
    """
    Consolidate all 'metrics.csv' files under input_dir into one DataFrame, including all task parameter columns.
    """
    csv_paths = find_all_metrics_csvs(input_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No metrics.csv found under {input_dir!r}")

    frames = read_csv_files(csv_paths)
    if not frames:
        raise FileNotFoundError(f"Failed to read any CSVs from: {csv_paths!r}")

    combined = pd.concat(frames, ignore_index=True)

    # Determine all columns that appear in any frame
    all_columns = set()
    for df in frames:
        all_columns.update(df.columns)

    # Define standard/base columns
    base_fieldnames = [
        "metric",
        "value",
        "task",
        "method",
        "num_simulations",
        "observation_idx",
    ]

    # All other columns are assumed to be task parameters or extra metadata
    task_param_columns = sorted([c for c in all_columns if c not in base_fieldnames])

    # Final column order: base + sorted task params
    final_columns = base_fieldnames + task_param_columns

    # Ensure all columns are present (add missing ones as NaN)
    for col in final_columns:
        if col not in combined.columns:
            combined[col] = pd.NA

    # Reorder columns
    combined = combined[final_columns]

    combined.to_csv(output_file, index=False)
    return combined

def main():
    args = parse_args()

    try:
        df = consolidate_metrics(input_dir=args.input_dir, output_file=args.output_file)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)

    print(f"Wrote {len(df)} rows to {args.output_file!r}")

if __name__ == "__main__":
    main()