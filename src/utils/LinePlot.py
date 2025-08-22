from typing import List, Tuple, Optional, Union

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator

import pandas as pd

from pathlib import Path

from src.utils.BasePlot import BasePlot


class LinePlot(BasePlot):
    """
        Creates a FacetGrid line plot from benchmark results.

        Produces a Seaborn FacetGrid where columns are methods and rows are
        (task, metric) pairs (supports multiple metrics per task). Subplot
        panels have fixed physical size (inches); the total figure size is
        derived from panel size plus margins/spacing to keep visuals comparable
        across different grid shapes.

        Args:
            data_sources (str | Path | list[str | Path]):
                One or more CSV files or directories to load (e.g. "metrics.csv").
            x (str):
                The name of the column to be plotted on the x-Axis.
                Default "num_simulations".
            base_directory (str | Path, optional):
                Root directory used to resolve relative paths and save outputs.
            filename (str, optional):
                Output filename (with or without extension). Defaults to
                "<ClassName>.png".
            plot_kwargs (dict, optional):
                Per‑line style overrides forwarded to ``sns.relplot``.
            row_order (list[str], optional):
                Desired facet row order. Accepts either task names (expanded to
                (task, metric) pairs found in the data) or explicit "task__metric"
                keys.
            col_order (list[str], optional):
                Desired facet column (method) order.
            title (str, optional):
                Figure title. Increases top margin if provided.
            log_x (bool, optional):
                Use logarithmic x‑axis. Defaults to True.
            err_style (str, optional):
                Seaborn error style (e.g., "bars", "band"). If omitted, defaults
                to "bars" with sensible ``err_kws``.

        Attributes:
            data (pd.DataFrame | None):
                Loaded DataFrame (after ``load_data``).
            fig (plt.Figure | None):
                Created figure (after ``create``).
            axes (list[plt.Axes] | None):
                Flat list of subplot axes (after ``create``).
            save_path (Path | None):
                Path where the figure was saved (after ``save``).

        Notes:
            - Rows are faceted by a combined key "task__metric" to support multiple
              metrics per task; if only one metric exists globally, a single super
              y‑label is used instead.
            - Panel size and margins are inch‑based; bounds are converted to
              figure‑relative coordinates for positioning headers and labels.
    """
    def __init__(
            self,
            data_sources: Union[str, Path, List[Union[str, Path]]],
            *,
            x: str = "num_simulations",

            base_directory: Optional[Union[str, Path]] = None,
            save_directory: Optional[Union[str, Path]] = None,
            filename: Optional[str] = None,
            plot_kwargs: Optional[dict] = None,

            row_order: Optional[List] = None,
            col_order: Optional[List] = None,
            title: Optional[str] = None,
            log_x: Optional[bool] = True,
            err_style: Optional[str] = None

    ):
        self.x = x

        super().__init__(
            data_sources,
            base_directory=base_directory,
            save_directory=save_directory,
            filename=filename,
            plot_kwargs=plot_kwargs
        )

        self.row_order = row_order
        self.col_order = col_order
        self.title = title
        self.log_x = log_x
        self.err_style = err_style

    def _plot(self, df: pd.DataFrame) -> Tuple[plt.Figure, list]:

        # 1) Constants
        # 1.1 Figure Level
        # Margins
        TOP_in = 1.5 if self.title else 1
        BOTTOM_in = 1
        LEFT_in = 1.1
        RIGHT_in = 0.3

        # Title
        TITLE_pt = 14
        TITLE_vpos_block = 0.75     # relative vertical position in TOP_in

        # Super Axis Labels
        SUPXLABEL_pt = 12
        SUPYLABEL_pt = 12
        SUPXLABEL_vpos_block = 0.3     # relative vertical position in BOTTOM_in
        SUPYLABEL_hpos_block = 0.3     # relative horizontal position in LEFT_in


        # 1.2 Grid Level
        # Subplot size and grid spacing
        SUBPLOT_HEIGHT_in = 1
        SUBPLOT_WIDTH_in = 1.5

        HSPACE_in = 0.618 * SUBPLOT_HEIGHT_in           # vertical padding between grid rows
        WSPACE_in = 0.1 * SUBPLOT_WIDTH_in              # horizontal padding between grid columns
        HSPACE_rel = HSPACE_in / SUBPLOT_HEIGHT_in      # HSPACE_in as a fraction of subplot height
        WSPACE_rel = WSPACE_in / SUBPLOT_WIDTH_in       # WSPACE_in as a fraction of subplot width

        # Task and Method labels
        TASK_BLOCK_in = 0.618 * HSPACE_in
        TASK_BOX_in = 0.618 * TASK_BLOCK_in
        TASK_LABEL_pt = 12
        TASK_BOX_vpos_block = 0.2       # relative vertical position in TASK_BLOCK_in
        TASK_LABEL_vpos_block = 0.25    # relative vertical position in TASK_BLOCK_in

        METHOD_BLOCK_in = 0.3
        METHOD_LABEL_pt = 11
        METHOD_LABEL_vpos_block = 0.4   # relative vertical position in METHOD_BLOCK_in


        # 1.3 Subplot Level
        # Plot styling
        # Use user-provided style/kwargs if given, else apply defaults
        DEFAULT_PLOT_KWARGS = {
            "dashes": False,
            "linewidth": 2,
            "color": "#023e96",

            "markers": True,
            "marker": "o",
            "markersize": 5,
            "markerfacecolor": "#011532",
            "markeredgewidth": 0.75,
            "markeredgecolor": "white",

            "zorder": 3
            }
        plot_kwargs = self.plot_kwargs or DEFAULT_PLOT_KWARGS

        # Error bar styling
        # Use user-provided style/kwargs if given, else apply defaults
        DEFAULT_ERR_KWS = {"linewidth": 1, "capsize": 1.5, "capthick": 1, "zorder": 2}
        ERR_STYLE = self.err_style or "bars"
        ERR_KWS = {} if self.err_style else DEFAULT_ERR_KWS




        # 2) Initialize grid and figure geometry
        # 2.1 Create combined row label ("task_metric") and map user-provided row_order if given
        # Ensures correct facet row order when tasks have multiple metrics.
        df = df.copy()
        df.sort_values(["task", "metric"], inplace=True)
        df["task_metric"] = df["task"] + "__" + df["metric"]

        tm_set = set(df["task_metric"])
        t_set = set(df["task"])

        if not self.row_order:  # None or []
            row_order = df["task_metric"].drop_duplicates().tolist()
        elif all(k in tm_set for k in self.row_order):  # user passed task_metric keys
            row_order = [k for k in self.row_order if k in tm_set]
        elif all(k in t_set for k in self.row_order):  # user passed task names
            row_order = []
            for t in self.row_order:
                row_order.extend(df.loc[df["task"] == t, "task_metric"].drop_duplicates().tolist())
        else:
            raise ValueError("row_order must be a list of 'task' names or 'task_metric' keys.")


        # 2.2 Create an initial FacetGrid
        grid = sns.relplot(
            data=df,

            x=self.x,
            y="value",
            row="task_metric",
            col="method",

            kind="line",
            estimator="mean",

            errorbar="sd",
            err_style=ERR_STYLE,
            err_kws=ERR_KWS,
            facet_kws={"sharey": True, "sharex": True},

            row_order=row_order,
            col_order=self.col_order or sorted(df['method'].unique()),

            **plot_kwargs,
        )


        # 2.3 Calculate figure size
        # Grid Dimensions
        num_rows, num_cols = grid.axes.shape

        # Figure Size
        # Set total figure size (in inches) based on fixed subplot dimensions and margins.
        # This ensures each subplot has the same physical size regardless of the number of rows/columns,
        # so comparisons between plots remain visually consistent.
        fig_height = TOP_in + SUBPLOT_HEIGHT_in + (num_rows - 1) * (HSPACE_in + SUBPLOT_HEIGHT_in) + BOTTOM_in
        fig_width = LEFT_in + SUBPLOT_WIDTH_in + (num_cols - 1) * (WSPACE_in + SUBPLOT_WIDTH_in) + RIGHT_in

        grid.fig.set_size_inches(fig_width, fig_height)     # Apply figure size (overrides Seaborne default)


        # 2.4 Figure-relative Layout Parameters
        # Grid bounds
        # Convert absolute margins (in inches) to figure-relative margins (0-1)
        top_bound = 1 - TOP_in / fig_height
        bottom_bound = BOTTOM_in / fig_height
        left_bound = LEFT_in / fig_width
        right_bound = 1 - RIGHT_in / fig_width

        # Grid center
        # Determine figure-relative coordinates of the vertical/horizontal center of the grid
        vertical_grid_center = (top_bound + bottom_bound) / 2
        horizontal_grid_center = (left_bound + right_bound) / 2

        # Grid size
        # ...
        grid_height = fig_height - (TOP_in + BOTTOM_in)
        grid_width = fig_width - (LEFT_in + RIGHT_in)

        # Y-coordinates (figure-relative) for the top edges of all subplot rows
        row_top = [1 - (TOP_in + i * (SUBPLOT_HEIGHT_in + HSPACE_in)) / fig_height for i in range(num_rows)]

        # Adjust Label size
        K = 0.6
        supx = self.x
        metrics = df["metric"].astype(str).unique()

        task_len = max(len(df.loc[df["task_metric"] == k, "task"].iloc[0]) for k in grid.row_names)
        method_len = max(len(str(m)) for m in grid.col_names)
        supx_len = len(supx)
        supy_len = len(metrics[0]) if len(metrics) == 1 else 0
        ylab_len = 0 if supy_len else max(len(df.loc[df["task_metric"] == k, "metric"].iloc[0]) for k in grid.row_names)

        cands = [
            72 * grid_width / (TASK_LABEL_pt * task_len * K),
            72 * SUBPLOT_WIDTH_in / (METHOD_LABEL_pt * method_len * K),
            72 * grid_width / (SUPXLABEL_pt * supx_len * K),
            *([72 * grid_height / (SUPYLABEL_pt * supy_len * K)] if supy_len else []),
            *([72 * SUBPLOT_HEIGHT_in / (SUPYLABEL_pt * ylab_len * K)] if ylab_len else []),
        ]
        s = min(1.0, *cands)

        task_label_pt = TASK_LABEL_pt * s
        method_label_pt = METHOD_LABEL_pt * s
        supxlabel_pt = SUPXLABEL_pt * s
        supylabel_pt = SUPYLABEL_pt * s


        # 3) Subplot Layout
        # 3.1 Axes Limits and Scales
        # Calculate x-axis limits
        # Ensure consistent visual padding to the left and right of the data range, regardless of x-axis scale
        x_min = df[self.x].min()
        x_max = df[self.x].max()

        if self.log_x:
            xlim = (x_min * 0.5, x_max * 2)
        else:
            xlim = (x_min - 0.2 * (x_max - x_min), x_max + 0.2 * (x_max - x_min))

        # Set y-axis limits
        # Fixed to ensure easy visual comparison between figures
        # Covers the theoretical metric range (0.5, 1.0) of all the currently supported metrics
        ylim = (0.45, 1.00)

        # Apply limits and scaling to all subplots
        for ax in grid.axes.flat:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if self.log_x:
                ax.set_xscale("log")


        # 3.2 Ticks and Tick Labels
        # Hide x-axis ticks and labels on all subplots except the bottom row
        for ax in grid.axes[:-1, :].flat:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

        # Hide y-axis ticks and labels on all subplots except the left column
        for ax in grid.axes[:, 1:].flat:
            ax.tick_params(axis="y", left=False, labelleft=False)

        # Remove minor tick labels and minor tick marks on x-axis
        for ax in grid.axes.flat:
            ax.tick_params(
                axis="x",
                which="minor",
                length=0,           # hide minor‐tick marks
                labelbottom=False   # hide minor‐tick labels
            )

        # Ensure y-axis major ticks at 0.1 increments
        for ax in grid.axes.flat:
            ax.yaxis.set_major_locator(MultipleLocator(0.1))


        # 3.3 Spines and Gridlines
        # Hide the bottom spine on all subplots except the bottom row
        for ax in grid.axes[:-1, :].flat:
            ax.spines["bottom"].set_visible(False)

        # Hide the left spine on all subplots except the left column
        for ax in grid.axes[:, 1:].flat:
            ax.spines["left"].set_visible(False)

        # Configure gridlines
        for ax in grid.axes.flat:
            ax.grid(False)                                      # Turn off the default grid
            ax.xaxis.grid(True, linestyle="--", linewidth=0.5)  # Vertical grid lines (dashed)
            ax.yaxis.grid(True, linestyle="-", linewidth=0.5)   # Horizontal grid lines (solid)




        # 4) Figure and Grid Layout
        # 4.1 Spacing
        grid.fig.subplots_adjust(
            top=top_bound,          # top margin
            bottom=bottom_bound,    # bottom margin
            left=left_bound,        # left margin
            right=right_bound,      # right margin

            hspace=HSPACE_rel,      # vertical spacing between rows
            wspace=WSPACE_rel       # horizontal spacing between columns
        )


        # 4.2 Row and Column Headers

        # 4.2.1 Add task names as row headers with background boxes
        # Calculate background box size
        box_width = (fig_width - (LEFT_in + RIGHT_in)) / fig_width
        box_height = TASK_BOX_in / fig_height

        # Calculate positions of the task label and background box
        task_block_pos = [top_bound + METHOD_BLOCK_in / fig_height] + row_top[1::]  # Special position for the top row
        task_label_pos = [y + TASK_LABEL_vpos_block * TASK_BLOCK_in / fig_height for y in task_block_pos]
        task_box_pos = [y + TASK_BOX_vpos_block * TASK_BLOCK_in / fig_height for y in task_block_pos]

        for i, task in enumerate(grid.row_names):
            # Draw background rectangle
            rect = FancyBboxPatch(
                (left_bound, task_box_pos[i]),
                box_width, box_height,
                boxstyle="round,pad=0",
                linewidth=0,                        # no border
                facecolor="#f0f0f0",
                transform=grid.fig.transFigure,     # position in figure-coordinates
                clip_on=False                       # allow drawing outside axes bounds
            )

            rect.set_in_layout(False)       # exclude from tight layout calculations
            grid.fig.add_artist(rect)       # add rectangle as a figure-level artist (not tied to any Axes)


            # Add task name label
            task_name = df.loc[df["task_metric"] == grid.row_names[i], "task"].iloc[0]

            grid.fig.text(
                horizontal_grid_center, task_label_pos[i],
                task_name,
                ha="center", va="bottom",
                fontsize=task_label_pt, weight="bold"
            )



        # 4.2.2 Add method names as column headers (only top row)
        # Calculate position of the method label
        method_block_pos = top_bound
        method_label_pos = method_block_pos + METHOD_LABEL_vpos_block * METHOD_BLOCK_in / fig_height

        for j, method in enumerate(grid.col_names):
            # Calculate the horizontal center of top-row Axes (fig coords)
            ax = grid.axes[0, j]
            pos = ax.get_position()
            x_center = (pos.x0 + pos.x1) / 2

            # Add method label centered above the column
            grid.fig.text(
                x_center, method_label_pos,
                str(method),
                ha="center", va="bottom",
                fontsize=method_label_pt, weight="medium", color="#222222",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#f0f0f0",
                    edgecolor="none"
                )
            )


        # 4.3 Global Labels and Title:
        # Remove subplot titles (set by Seaborn by default)
        grid.set_titles("")

        # Add a super x-axis label
        supxlabel_pos = (SUPXLABEL_vpos_block * BOTTOM_in) / fig_height

        for ax in grid.axes.flat:
            ax.set_xlabel(None)

        grid.fig.supxlabel(
            self.x,
            x=horizontal_grid_center,
            y=supxlabel_pos,
            ha="center",
            va="center",
            fontsize=supxlabel_pt
        )


        # Add a super y-axis label if applicable
        unique_metrics = df["metric"].unique()

        if len(unique_metrics) == 1 and len(df["metric"]) != 1:
            # Only one metric; add a single super y-label
            for ax in grid.axes.flat:
                ax.set_ylabel(None)

            supy_pos = (SUPYLABEL_hpos_block * LEFT_in) / fig_width

            grid.fig.supylabel(unique_metrics[0],
                               x=supy_pos,
                               y=vertical_grid_center,
                               rotation="vertical",
                               ha="center",
                               va="center",
                               fontsize=supylabel_pt)

        else:
            # Multiple metrics; add row-specific y-labels on the leftmost column
            for i, row_axes in enumerate(grid.axes):
                metric_name = df.loc[df["task_metric"] == grid.row_names[i], "metric"].iloc[0]
                row_axes[0].set_ylabel(metric_name)


        # Add a super title for the entire figure
        if self.title:
            title_pos = top_bound + (TOP_in * TITLE_vpos_block) / fig_height

            grid.fig.suptitle(
                self.title,
                x=horizontal_grid_center,
                y=title_pos,
                ha="center",
                va="center",
                fontsize=TITLE_pt,
                fontweight="bold",
            )



        # 5) Output
        fig = grid.fig
        axes = [ax for row in grid.axes for ax in row]

        return fig, axes
