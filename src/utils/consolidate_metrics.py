"""
consolidate_metrics

Combine `metrics.csv` files under a given input dir into a single CSV and save it at a given output file.

The script will:
    1. Glob for `sims_*/obs_*/metrics.csv` under `input_dir`
    2. Read each file into a DataFrame
    3. Concatenate them into one combined DataFrame
    4. Validate columns
    5. Write the result to `output_file`


Usage:
    python consolidate_metrics.py --input_dir <base_directory> --output_file <metrics_all.csv>
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
from src.utils.csv_utils import gather_csv_files, read_csv_files


def parse_args():
    """Parse and return command-line arguments."""
    p = argparse.ArgumentParser(
        description="Combine all metrics.csv into one CSV."
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Base directory containing sims_*/obs_*/metrics.csv"
    )
    p.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to write the combined CSV (e.g. metrics_all.csv)"
    )
    return p.parse_args()


def consolidate(input_dir: Path, output_file: Path) -> pd.DataFrame:
    """
    Glob all 'metrics.csv' files under `input_dir`, concatenate them into a single DataFrame,
    write that DataFrame to `output_file`, and return it.

    Args:
        input_dir: Base directory under which to glob for sims_*/obs_*/metrics.csv
        output_file: Where to write the combined CSV

    Returns:
        The concatenated DataFrame.

    Raises:
        FileNotFoundError: If no metrics.csv files are found.
        ValueError: If base fieldnames are missing/mis-ordered or contain missing values.
    """
    # 1) gather metrics.csv files
    pattern = input_dir / "sims_*" / "obs_*" / "metrics.csv"
    csv_paths = gather_csv_files(data_sources=str(pattern), base_directory=input_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No metrics.csv under {input_dir!r}")

    # 2) read all .csv files
    frames = read_csv_files(csv_paths)

    # 3) concatenate into one combined dataframe
    combined = pd.concat(frames, ignore_index=True)

    # 4) Validate columns
    # 4a) Assert the base fieldnames exist and metadata columns are sorted
    base_fieldnames = [
        "metric",
        "value",
        "task",
        "method",
        "num_simulations",
        "observation_idx",
    ]
    # Required columns must all be present, in that order
    cols = list(combined.columns)
    if cols[: len(base_fieldnames)] != base_fieldnames:
        raise ValueError(
            f"Wrong CSV structure: expected first columns {base_fieldnames!r}, "
            f"but got {cols[: len(base_fieldnames)]!r}"
        )

    # Any extra (metadata) cols must come after and sorted lexicographically
    extra = cols[len(base_fieldnames):]
    if extra != sorted(extra):
        raise ValueError(
            f"Unexpected or mis-ordered metadata columns: {extra!r}. "
            f"Should be sorted."
        )

    # 4b) Assert none of the base fieldnames columns contain missing values
    required = ["metric", "value", "task", "method", "num_simulations", "observation_idx"]
    missing = combined[required].isnull().sum()
    if missing.any():
        raise ValueError(f"Missing data in required columns:\n{missing[missing > 0]}")

    # 5) write out
    combined.to_csv(output_file, index=False)

    return combined


def main():
    args = parse_args()

    try:
        df = consolidate(args.input_dir, args.output_file)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)

    print(f"Wrote {len(df)} rows to {args.output_file!r}")


if __name__ == "__main__":
    main()
