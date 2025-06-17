from pathlib import Path
from itertools import count
from typing import Literal, Tuple


def ensure_directory(directory: Path | str) -> None:
    """
    Ensure that the directory exists.

    If it does not yet exist, the directory and any missing parents will be created.
    Otherwise, no changes are made.

    Args:
        directory (Path | str): Path to the directory to ensure.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)


def unique_path(path: Path, *, sep: str = "__") -> Path:
    """
    Generate a unique filename within the same directory by appending a running ID if needed,
    to make sure the path is unique and doesn't collide with existing paths.

    If the given path does not exist, return it unchanged.
    Otherwise, append a numeric running ID to the filename stem to make it unique within the same directory.
    (e.g. /output/results/gaussian_linear__nNLE.csv ⇾
          /output/results/gaussian_linear__nNLE__1.csv).

    Args:
        path (Path): Desired file path.
        sep (str, optional): Separator between the original file stem and the running ID. Defaults to "__".

    Returns:
        Path: A generated Path with a unique filename within the same directory.
    """
    if not path.exists():
        return path

    # Split the filename into its stem and its extension (e.g., results.csv -> stem = "results", extension = ".csv")
    stem, extension = path.stem, path.suffix

    for n in count(1):
        # Creates potentially unique filename by adding/ increasing the running ID 'n'
        candidate = path.with_name(f"{stem}{sep}{n}{extension}")
        if not candidate.exists():
            return candidate  # Return if unique, otherwise increase the running ID 'n' and try again


def create_unique_path(
    save_directory: str | Path,
    stem: str,
    *,
    extension: str = ".csv",
    sep: str = "__",
) -> Path:
    """
    Ensure the save directory exists and generate a unique filename within the same directory.

    Creates the directory `save_directory` and any missing parents if it doesn't exist yet.
    Then combines the given `stem` and `extension` to form a filename.
    If a file with that name already exists, appends a numeric running ID to avoid collisions.
    (e.g. /output/results/gaussian_linear__nNLE.csv ⇾
          /output/results/gaussian_linear__nNLE__1.csv).

    Args:
        save_directory (str | Path): Directory where the file will be saved.
        stem (str): Stem of the desired filename (without extension).
        extension (str, optional): File extension (including leading dot). Defaults to ".csv".
        sep (str, optional): Separator between stem and extension. Defaults to "__".

    Returns:
        Path: A created Path with a unique filename within the same directory.
    """
    directory = Path(save_directory)
    ensure_dir(directory)
    return unique_path(directory / f"{stem}{extension}", sep=sep)


def resolve_write_mode(path: Path, write_mode: Literal["write", "append"] = "append") -> Tuple[str, bool]:
    """
    Determine write mode and whether to write the header row or not.

        - "write": always create a new file (write mode "w"), write header.
        - "append": append if the file exists (write mode "a"), don't write header; Otherwise, behave like "write".

    Args:
        path (Path): The target file path.
        write_mode (WriteMode): "write" or "append".
            Defaults to "append".

    Returns:
        Tuple[str, bool]: (mode, write_header), where `mode` is "w" or "a",
                          and `write_header` a boolean that indicates if a header row should be written.
    """
    if write_mode == "write":
        return "w", True  # Write a new file, and create a header
    if write_mode == "append":
        if path.exists():
            return "a", False  # Append to an existing file, and don't create a header
        return "w", True  # Write a new file, and create a header
    raise ValueError(f"Unknown file_mode: {write_mode!r}")
