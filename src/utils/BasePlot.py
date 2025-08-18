from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Iterable, Optional, Tuple, Union, Set

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.file_utils import ensure_directory, unique_path
from src.utils.csv_utils import gather_csv_files, read_csv_files, ensure_columns


class BasePlot(ABC):
    """
    BasePlot handles loading CSV files and saving figures; subclasses implement the plotting logic.

    Args:
        data_sources (str | Path | list[str|Path]): Sources to gather benchmark results from (e.g.: metrics.csv)
        base_directory (str | Path, optional): Root directory for loading and saving; defaults to cwd.
        filename (str, optional): Custom output filename (stem or with extension).
            Defaults to "<ClassName>.png" where ClassName is the name of the subclass.
        **plot_kwargs: Passed to plotting calls in subclass, e.g. {"marker": "o", "linestyle": "--"}.

    Attributes:
        data (pd.DataFrame | None): Set by load_data(); holds the loaded DataFrame.
        fig (plt.Figure | None): Set by create(); holds the produced figure.
        axes (list[plt.Axes] | None): Set by create(); holds the flat list of axes.
        save_path (Path | None): Set by save(); holds path where the figure was saved.

    Subclasses must implement:
        def _plot(self, df: pd.DataFrame, **plot_kwargs) -> (Figure, list[Axes])
    """
    def __init__(
            self,
            data_sources: Union[str, Path, Iterable[Union[str, Path]]],
            *,
            base_directory: Optional[Union[str, Path]] = None,
            save_directory: Optional[Union[str, Path]] = None,
            filename: Optional[str] = None,

            plot_kwargs: Optional[dict] = None,
    ):
        # Normalization:
        # Normalize data_sources to a list of Path objects
        if isinstance(data_sources, (str, Path)):
            self.data_sources = [Path(data_sources)]
        else:
            self.data_sources = [Path(p) for p in data_sources]

        # Normalize base_directory to an absolute Path
        if base_directory is None:
            self.base_directory = Path.cwd().resolve()
        else:
            self.base_directory = Path(base_directory).resolve()

        # Normalize save_directory to an absolute Path
        if save_directory is None:
            self.save_directory = self.base_directory / "outputs" / "plots"
        else:
            sd = Path(save_directory)
            if not sd.is_absolute():
                sd = self.base_directory / sd
            self.save_directory = sd.resolve()

        # Normalize filename and extension
        if filename is None:
            self._filename_is_default = True
            self.stem = self.__class__.__name__     # name of the Subclass/Plot (e.g. "LinePlot")
            self.extension = "png"
        else:
            self._filename_is_default = False
            f = Path(filename)
            self.stem = f.stem
            # Use extension from filename if present, otherwise fallback to default extension "png"
            self.extension = f.suffix.lstrip('.') if f.suffix else "png"



        # Kwargs for plotting a CSV file (e.g.,{"marker": "o", "linestyle": "--"})
        self.plot_kwargs = plot_kwargs or {}

        # Output attributes
        self.data: Optional[pd.DataFrame] = None    # Output of load_data()
        self.fig: Optional[plt.Figure] = None       # Output of plot()
        self.axes: Optional[List[plt.Axes]] = None  # Output of plot()
        self.save_path: Optional[Path] = None       # Output of save()


    def run(self) -> Path:
        """
        Run the full pipeline: load the data, create the plot, save the figure and return the save path.

        Use run() for the standard end-to-end flow. If you need to customize the plot before saving, do:
            >>> plotter = BasePlot(...)
            >>> df = plotter.load_data()
            >>> fig, axes = plotter.plot(df)
            >>> # CUSTOMIZATION, e.g.:
            >>> axes[0].set_title("Custom Title") # etc.
            >>> save_path = plotter.save(fig)

        Returns:
            Path: The save path to the plotted figure.
        """
        df = self.load_data()
        fig, axes = self.plot(df)
        save_path = self.save(fig)
        return save_path


    def load_data(self) -> pd.DataFrame:
        """
        Load CSV files from self.data_sources as a pd.DataFrame and concatenate them into one combined pd.DataFrame.

        Returns:
            pd.DataFrame: The combined DataFrame of all successfully read CSV files.

        Raises:
            FileNotFoundError: If no CSV files are found for the given data_sources.
            ValueError: If CSV files are found but all reads fail.
        """
        # Gather all CSV files from the data_sources
        csv_paths = gather_csv_files(data_sources=self.data_sources, base_directory=self.base_directory)
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found for any of {self.data_sources!r}")

        # Read each CSV file into a single DataFrame
        frames = read_csv_files(csv_paths)
        if not frames:
            raise ValueError(f"All CSV reads failed for {self.data_sources!r}")

        # Concatenate frames into one combined DataFrame
        combined = pd.concat(frames, ignore_index=True)
        self.data = combined  # update self.data attribute for future potential use

        # 4) Validate and reorder columns
        base_fieldnames = [
            "metric",
            "value",
            "task",
            "method",
            "num_simulations",
            "observation_idx",
        ]

        combined = ensure_columns(combined, base_fieldnames)

        # Report how many rows/files were loaded
        print(f"Loaded {len(combined)} rows from {len(frames)} files.")
        return combined


    def plot(self, df: pd.DataFrame) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Generate a plot using the Subclass's plotting logic.

        Args:
            df (pd.DataFrame): DataFrame to plot, typically obtained from `load_data()`.

        Returns:
            Tuple[plt.Figure, List[plt.Axes]]: The Figure object and a flat list of Axes.
        """
        # Call Subclass logic for plotting
        fig, axes = self._plot(df)

        # store attributes
        self.fig = fig
        self.axes = axes
        return fig, axes


    def save(self, fig: plt.Figure) -> Path:
        """
            Save the figure to disk.

            The final save path has the structure:
                base_directory / save_dir / filename.extension

            - base_directory: absolute root directory (default: cwd)
            - save_dir: subdirectory inside base_directory (default: 'outputs/plots')
            - filename: custom stem or class name (unique if not provided)

            Args:
                fig (plt.Figure): The matplotlib Figure to save.

            Returns:
                Path: The path where the figure was saved.
            """
        # 1) Determine the save path:
        filename = f"{self.stem}.{self.extension}"
        save_path = self.save_directory / filename

        # If the filename was auto-generated (default),
        # make it unique by appending a running ID to avoid accidental overwriting
        if self._filename_is_default:
            save_path = unique_path(save_path)

        # Ensure the save directory (and any missing parents) exists
        ensure_directory(save_path.parent)


        # 2) Save the figure (under the save path)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

        # 3) Store and report the save path
        self.save_path = save_path
        print(f"Saved figure ➜ {save_path}")
        return save_path


    @abstractmethod
    def _plot(self, df: pd.DataFrame) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Subclasses implement plotting logic here.
        Must return (fig, axes), where axes is a flat list of Axes.
        """
        ...
