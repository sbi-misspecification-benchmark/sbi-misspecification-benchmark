import csv
import glob
from pathlib import Path
from typing import Iterable, List, Literal, Sequence, Tuple, Union

import pandas as pd


def get_csv_header(path: Path) -> List[str]:
    """
    Read the header row from a CSV file.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        List[str]: The list of column names from the first row.

    Raises:
        ValueError: If the file is empty and therefore, no header row is found.
    """
    with path.open(newline="") as f:
        try:
            return next(csv.reader(f))
        except StopIteration:
            raise ValueError(f"{path} is empty. No header found")


def assert_csv_header_matches(path: Path, expected_header: Sequence[str]) -> None:
    """
    Check whether a .csv file's header matches an expected header sequence.

    Args:
        path (Path): Path of the .csv file.
        expected_header (Sequence[str]): The expected sequence of column names.

    Raises:
        FileNotFoundError: If the .csv file is not found.
        ValueError: If the file is not a .csv file;
                    If the .csv file's header differs from the expected header sequence.
    """
    # Assert .csv file exists:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, but got: {path}")

    # Assert .csv header matches the expected header sequence
    header = get_csv_header(path)
    if header != list(expected_header):
        raise ValueError(
            f"CSV header mismatch in {path!r}:\n"
            f"  got: {header!r}\n"
            f"  expected: {list(expected_header)!r}"
        )


def resolve_file_mode(path: Path, file_mode: Literal["write", "append"]) -> Tuple[str, bool]:
    """
    Determine how to open a file and whether to write a header.

    Args:
        path (Path): Path to the target file.
        file_mode (str): Either "write" or "append".

    Returns:
        Tuple[str, bool]: A tuple (mode, write_header).
            If file_mode == "write": returns ("w", True).
            If file_mode == "append" and the file exists: returns ("a", False).
            If file_mode == "append" and the file does not exist: returns ("w", True).

    Raises:
        ValueError: If file_mode is not "write" or "append".
    """
    if file_mode == "write":
        return "w", True  # Write a new file, and create a header

    elif file_mode == "append":
        if path.exists():
            return "a", False  # Append to an existing file, and don't create a header
        return "w", True  # Write a new file, and create a header

    else:
        raise ValueError(f"Unknown file_mode: {file_mode!r}")


def gather_csv_files(
        data_sources: Union[str, Path, Iterable[Union[str, Path]]],
        base_directory: Path
) -> List[Path]:
    """
    Resolve self.data_sources into a set of CSV Paths.
    - If a source is an existing .csv file, include it.
    - If a source is a directory, include all .csv under it.
    - If a source is a glob pattern, glob under base_directory if relative, or from root if absolute.

    Args:
        data_sources (str | Path | Iterable[Union[str, Path]]): Sources to gather absolute .csv file paths from.
        base_directory (Path): Absolute root directory for relative glob patterns

    Returns:
        List[Path]: List of absolute .csv file paths.
    """

    # Normalize sources
    if isinstance(data_sources, (str, Path)):
        data_sources = [Path(data_sources)]
    elif isinstance(data_sources, Iterable):
        data_sources = [Path(p) for p in data_sources]
    else:
        raise TypeError(f"data_sources must be str, Path, or an iterable thereof; got {type(data_sources).__name__}")


    # Matching .csv file paths of all sources
    csv_paths: set[Path] = set()

    for source in data_sources:
        source_path = Path(source)  # Source_path used for a direct file and directory
        if not source_path.is_absolute():
            source_path = base_directory / source_path
        matched: set[Path] = set()  # Matching .csv file paths for the source of this loop

        # 1) Direct file
        if source_path.is_file() and source_path.suffix.lower() == ".csv":
            matched.add(source_path)

        # 2) Directory
        elif source_path.is_dir():
            matched.update(source_path.rglob("*.csv"))

        # 3) Glob pattern
        elif glob.has_magic(str(source_path)):
            # Normalize glob pattern (turn relative patterns to absolute patterns)
            pattern = str(source_path) if source_path.is_absolute() else str(base_directory / source_path)
            # Gather all .csv files that match the glob pattern
            for match_str in glob.glob(pattern, recursive=True):
                match_path = Path(match_str)
                if match_path.is_file() and match_path.suffix.lower() == ".csv":
                    matched.add(match_path)

        # 4) Neither a file, directory, nor a glob pattern
        else:
            print(f"No .csv file, directory, or glob pattern matched for source: {source!r}")

        # Inform if a source did not match .csv file(s)
        if not matched:
            print(f"No .csv files found for source: {source!r}")
        else:
            csv_paths.update(matched)

    return list(csv_paths)


def read_csv_files(paths: List[Path]) -> list[pd.DataFrame]:
    """
    Read each .csv file in `paths` into a pandas DataFrame, and return the list of DataFrames.

    Args:
        paths (List[Path]): List of absolute .csv file paths.

    Returns:
        List[pd.DataFrame]: A list of DataFrames for each CSV successfully read.
            If reading a file fails, that file is skipped; the returned list may be empty.
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as e:
            print(f"Failed to read {path!r}: {e}")
    return frames


def ensure_columns(
        df: pd.DataFrame,
        required_columns: Sequence[str],
        *,
        sort_extra: bool = True
) -> pd.DataFrame:
    """
    Validate and reorder columns of DataFrame 'df'.

    Ensure all 'required_columns' exist in 'df' and do not contain missing values.
    Optionally sort any extra columns beyond the 'required_columns'.
    Return a new DataFrame with columns ordered as: [*required_columns, *extra_columns].

    Args:
        df (pd.DataFrame): The DataFrame to validate and reorder.
        required_columns (Sequence[str]): Names of columns that must exist and cannot contain missing values.
        sort_extra (bool): If True, any extra columns (beyond the 'required_columns') will be sorted lexicographically.

    Returns:
        pd.DataFrame: A new DataFrame with the validated and reordered columns.

    Raises:
        ValueError: If any 'required_columns' are missing or contain missing values.
    """
    # Ensure all required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure no required column has missing values (NaNs)
    missing_value_cols = [col for col in required_columns if df[col].isnull().any()]
    if missing_value_cols:
        raise ValueError(f"Required columns contain missing values: {missing_value_cols}")

    # Identify extra columns beyond the required columns, and optionally sort them lexicographically
    extra_columns = [col for col in df.columns if col not in required_columns]
    if sort_extra:
        extra_columns.sort()

    # Build column order and reindex the df to match it
    column_order = list(required_columns) + extra_columns
    ordered_df = df.reindex(columns=column_order)

    return ordered_df
