"""
Consolidate all 'metrics.csv' files (of the same task and method) across simulations into a single DataFrame.

This will:
1. Gather all 'metrics.csv' files from all simulation subfolders 'sim_*' at given input directory `input_dir`
2. Read and concatenate them into a single DataFrame.
3. Validate and reorder columns.
4. Write the resulting DataFrame to 'output_file' as a .csv file, then return it.

Usage:
    python consolidate_metrics.py --input_dir <base_directory> --output_file <metrics_all.csv>
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

import pandas as pd
from src.utils.csv_utils import gather_csv_files, read_csv_files, ensure_columns


def parse_args():
    """Parse and return command-line arguments."""
    p = argparse.ArgumentParser(
        description="Consolidate all 'metrics.csv' files (of the same task and method) across simulations into a "
                    "single DataFrame."
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Base directory containing sims_*/metrics.csv files to consolidate"
    )
    p.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="File path where to write the consolidated .csv file (e.g. metrics_all.csv)"
    )
    return p.parse_args()


def consolidate_metrics(input_dir: Path, output_file: Path) -> pd.DataFrame:
    """
    Consolidate 'metrics.csv' files (of the same task and method) across simulations into one DataFrame.

    1. Gather all 'metrics.csv' files from all simulation subfolders 'sim_*' at given input directory `input_dir`
    2. Read and concatenate them into a single DataFrame.
    3. Validate and reorder columns.
    4. Write the resulting DataFrame to 'output_file' as a CSV file, then return it.

    Args:
        input_dir: Base directory under which to glob for sims_*/metrics.csv
        output_file: File path where to write the consolidated .csv file.

    Returns:
        The consolidated DataFrame.

    Raises:
        FileNotFoundError: If no metrics.csv files can be read or found.
        ValueError: If base fieldnames are missing or contain missing values.
    """
    # 1) Merge CSV files
    # 1.1) Gather all 'metrics.csv' files from all simulation subfolders 'sim_*' at given input directory `input_dir`
    csv_paths = gather_csv_files(data_sources="sims_*/metrics.csv", base_directory=input_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No metrics.csv under {input_dir!r}")

    # 1.2) Read all .csv files
    frames = read_csv_files(csv_paths)
    if not frames:
        raise FileNotFoundError(f"Failed to read any CSVs from: {csv_paths!r}")

    # 1.3) Concatenate frames into one combined DataFrame
    combined = pd.concat(frames, ignore_index=True)


    # 2) Validate and reorder columns
    base_fieldnames = [
        "metric",
        "value",
        "task",
        "method",
        "num_simulations",
        "observation_idx",
    ]
    combined = ensure_columns(combined, base_fieldnames)


    # 3) Write out at given output directory 'output_file'
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
