from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Iterable, Optional, Tuple, Union, Set

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.file_utils import ensure_directory, unique_path
from src.utils.csv_utils import gather_csv_files, read_csv_files


class BasePlot(ABC):
    """
    BasePlot handles loading .csv files and saving figures; subclasses implement the plotting logic.

    Args:
        data_sources (str | Path | list[str|Path]): Sources to gather benchmark results from.
        base_directory (str | Path, optional): Absolute root directory for loading and saving; defaults to cwd.
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
            filename: Optional[str] = None,
            extension: Optional[str] = "png",
            plot_kwargs: Optional[dict] = None,
    ):
        # Normalize data_sources to a list of Path objects
        if isinstance(data_sources, (str, Path)):
            ds = [Path(data_sources)]
        else:
            ds = [Path(p) for p in data_sources]

        self.data_sources = ds


        # Normalize base_directory to an absolute Path
        if base_directory is None:
            bd = Path.cwd()
        else:
            bd = Path(base_directory)
            if not bd.is_absolute():
                raise ValueError(f"base_directory must be absolute, got {base_directory!r}")

        self.base_directory = bd  # absolute Path
        self.filename = filename
        self.extension = extension


        # Kwargs for plotting a .csv file (e.g.,{"marker": "o", "linestyle": "--"} )
        self.plot_kwargs = plot_kwargs or {}

        # Output attributes
        self.data: Optional[pd.DataFrame] = None    # Output of load_data()
        self.fig: Optional[plt.Figure] = None       # Output of plot()
        self.axes: Optional[List[plt.Axes]] = None  # Output of plot()
        self.save_path: Optional[Path] = None       # Output of save()


    def run(self) -> Path:
        """
        Run the full pipeline: load data, create the plot, save the figure and return the save path.

        Use run() for the standard end-to-end flow. If you need to customize the plot before saving, do:
            >>> plotter = BasePlot(...)
            >>> df = plotter.load_data()
            >>> fig, axes = plotter.plot(df)
            >>> # CUSTOMIZATION, e.g.:
            >>> axes[0].set_title("Custom Title")
            >>> save_path = plotter.save(fig)

        Returns:
            Path: The save path to the plotted figure.
        """
        df = self.load_data()
        fig, ax = self.plot(df)
        save_path = self.save(fig)
        return save_path


    def load_data(self) -> pd.DataFrame:
        """
        Load .csv files from self.data_sources as a pd.DataFrame and concatenate them into one combined pd.DataFrame.

        Returns:
            pd.DataFrame: The combined DataFrame of all successfully read .csv files.

        Raises:
            FileNotFoundError: If no CSV files are found for the given data_sources.
            ValueError: If CSV files are found but all reads fail.
        """
        # Gather all CSV file paths matching the data_sources (file, dir, or glob under base_directory)
        csv_paths = gather_csv_files(data_sources=self.data_sources, base_directory=self.base_directory)
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found for any of {self.data_sources!r}")

        # Read each CSV into a single dataframe, tagging with "__source__"
        dataframes = read_csv_files(csv_paths)
        if not dataframes:
            raise ValueError(f"All CSV reads failed for {self.data_sources!r}")

        # Concatenate into one combined dataframe
        combined = pd.concat(dataframes, ignore_index=True)
        self.data = combined  # update self.data attribute for future potential use

        # Report how many rows/files were loaded
        print(f"Loaded {len(combined)} rows from {len(dataframes)} files.")
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
        fig, axes_list = self._plot(df, **self.plot_kwargs)

        # store attributes
        self.fig = fig
        self.axes = axes_list
        return fig, axes_list


    def save(self, fig: plt.Figure) -> Path:

        # Determine the save path:
        if self.filename:
            stem = Path(self.filename).stem  # Gets the stem, regardless of whether the filename had an extension or not
            save_path = (self.base_directory
                         / "outputs"
                         / "plots"
                         / f"{stem}.{self.extension}")
        else:
            stem = self.__class__.__name__  # Name of the Subclass
            desired_path = (self.base_directory
                            / "outputs"
                            / "plots"
                            / f"{stem}.{self.extension}")
            save_path = unique_path(desired_path)  # add a running ID to the stem if the desired path already exists

        # Ensure the save directory (and any missing parents) exists
        ensure_directory(save_path.parent)


        # Save the figure to the save path and close it
        fig.savefig(save_path)
        plt.close(fig)
        self.save_path = save_path


        # Confirm that the figure was saved
        print(f"Saved figure ➜ {save_path}")
        return save_path


    @abstractmethod
    def _plot(self, df: pd.DataFrame, **plot_kwargs) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Subclasses implement plotting here, using df and plot_kwargs (styling).
        Must return (fig, axes_list), where axes_list is a flat list of Axes.
        """
        ...
