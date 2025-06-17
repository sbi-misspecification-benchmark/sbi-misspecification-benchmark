import csv
from pathlib import Path
from typing import Sequence, List, Mapping, Any


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
    Check whether a CSV file's header matches an expected header sequence.

    Args:
        path (Path): Path to the CSV file.
        expected_header (Sequence[str]): The expected list of column names.

    Raises:
        ValueError: If the file's actual header differs from 'expected_header'.
    """
    header = get_csv_header(path)
    if header != list(expected_header):
        raise ValueError(
            f"CSV header mismatch in {path}:\n"
            f"got: {header}\n"
            f"expected: {list(expected_header)}"
        )


def merge_csv_rows(
    folder: Path,
    *,
    pattern: str = "*.csv",
    expected_header: Sequence[str] | None = None,
) -> List[Mapping[str, Any]]:
    """
    Merge rows from multiple CSV files in a folder into a single list of dicts.

    Args:
        folder (Path): Folder directory containing CSV files to merge.
        pattern (str, optional): Glob pattern to select files.
            Defaults to "*.csv".
        expected_header (Sequence[str], optional): If provided, validate each file's
            header against this sequence before reading.

    Returns:
        List[Mapping[str, Any]]: A list of row dictionaries from all CSVs.
    """
    rows: List[Mapping[str, Any]] = []
    for csv_path in sorted(folder.glob(pattern)):
        if expected_header is not None:
            assert_csv_header_matches(csv_path, expected_header)
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows
