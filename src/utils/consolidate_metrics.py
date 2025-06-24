"""
consolidate_metrics.py

Scan for all sim_*/obs_*/metrics.csv under <input_dir>, and merge them to one combined csv file.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from src.utils.csv_utils import gather_csv_files, read_csv_files


def parse_args():
    p = argparse.ArgumentParser(
        description="Combine all metrics.csv into one unified CSV."
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Base dir containing sims_*/obs_*/metrics.csv"
    )
    p.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to write the combined CSV (e.g. metrics_all.csv)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) gather metrics.csv files
    pattern = args.input_dir / "sims_*" / "obs_*" / "metrics.csv"
    csv_paths = gather_csv_files(data_sources=str(pattern), base_directory=args.input_dir)
    if not csv_paths:
        print(f"No metrics.csv found under {args.input_dir!r}", file=sys.stderr)
        sys.exit(1)

    # 2) read all .csv files
    frames = read_csv_files(csv_paths)

    # 3) concatenate into one combined dataframe
    combined = pd.concat(frames, ignore_index=True)

    # 4) write out
    combined.to_csv(args.output_file, index=False)
    print(f"Wrote {len(combined)} rows to {args.output_file!r}")


if __name__ == "__main__":
    main()