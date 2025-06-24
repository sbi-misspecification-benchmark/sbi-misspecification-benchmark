from itertools import count
from pathlib import Path


def ensure_directory(directory: Path | str) -> None:
    """
    Ensure that the directory and its parents exist.


    Args:
        directory (Path | str): Directory to ensure.
    """
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)


def unique_path(path: Path, *, sep: str = "_") -> Path:
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
    directory: str | Path,
    stem: str,
    *,
    extension: str,
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
        directory (str | Path): Directory where the file will be saved.
        stem (str): Stem of the desired filename (without extension).
        extension (str, optional): File extension (including leading dot). Defaults to ".csv".
        sep (str, optional): Separator between stem and extension. Defaults to "__".

    Returns:
        Path: A created Path with a unique filename within the same directory.
    """
    directory_path = Path(directory)
    ensure_directory(directory_path)
    return unique_path(directory_path / f"{stem}{extension}", sep=sep)
