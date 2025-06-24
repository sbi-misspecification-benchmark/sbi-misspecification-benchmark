from typing import List, Tuple, Optional, Union
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.utils.BasePlot import BasePlot


class LinePlot(BasePlot):
    def __init__(
            self,
            data_sources: Union[str, Path, List[Union[str, Path]]],
            *,
            base_directory: Union[str, Path] = None,
            filename: Optional[str] = None,
            plot_kwargs: Optional[dict] = None,

            x_col: str,
            y_col: str,
            method_col: str,
            obs_col: str,
            task_col: str,
            metric_col: str,
            row_order: Optional[List] = None,
            col_order: Optional[List] = None,
            log_x: bool = True,
            height: float = 3,
            aspect: float = 1.2,
            sharey: bool = True,
            sharex: bool = False,
            title: Optional[str] = None,
    ):
        super().__init__(
            data_sources,
            base_directory=base_directory,
            filename=filename,
            plot_kwargs=plot_kwargs,
        )
        self.x_col = x_col
        self.y_col = y_col
        self.method_col = method_col
        self.obs_col = obs_col
        self.task_col = task_col
        self.metric_col = metric_col
        self.row_order = row_order
        self.col_order = col_order
        self.log_x = log_x
        self.height = height
        self.aspect = aspect
        self.sharey = sharey
        self.sharex = sharex
        self.title = title

    def _plot(self, df: pd.DataFrame, **plot_kwargs) -> Tuple[plt.Figure, list]:
        # --- 1) Validate required columns ---
        for col in (self.x_col, self.y_col, self.method_col, self.obs_col, self.task_col, self.metric_col):
            if col not in df.columns:
                raise ValueError(f"Missing required column {col!r}")

        # --- 2) Style defaults ---
        plot_kwargs = plot_kwargs or {"linewidth": 1.2, "marker": "o"}

        # --- 3) Create the FacetGrid ---
        grid = sns.relplot(
            data=df,
            x=self.x_col, y=self.y_col,
            row=self.task_col,
            col=self.method_col,
            kind="line",
            estimator="mean",
            errorbar="sd",
            markers=True,
            dashes=False,
            height=self.height,
            aspect=self.aspect,
            facet_kws={"sharey": self.sharey, "sharex": self.sharex, "margin_titles": True},
            row_order=self.row_order,
            col_order=self.col_order,
            **plot_kwargs,
        )

        # --- Layout adjustments ---
        grid.fig.subplots_adjust(top=0.90, hspace=0.6)
        # Move method titles further up
        for ax in grid.axes[0, :]:
            ax.title.set_y(1.10)
        grid.set_titles(row_template="", col_template="{col_name}")

        # --- Clean up individual subplots ---
        axes = grid.axes
        for i, row_axes in enumerate(axes):
            for j, ax in enumerate(row_axes):
                ax.grid(True, linestyle="--", linewidth=0.5)
                # only show y-label on first column
                if j > 0:
                    ax.set_ylabel("")

        # --- Centered row titles above each row ---
        tasks = grid.row_names
        for i, task in enumerate(tasks):
            ax_l = axes[i, 0];
            ax_r = axes[i, -1]
            b_l = ax_l.get_position();
            b_r = ax_r.get_position()
            x_c = (b_l.x0 + b_r.x1) / 2;
            y_t = b_l.y1 + 0.03
            grid.fig.text(x_c, y_t, str(task), ha="center", va="bottom", weight="bold")

        # --- Shared axis labels and figure title ---
        # Determine y-label dynamically from metric_col
        unique_metrics = df[self.metric_col].unique()
        if len(unique_metrics) == 1:
            y_label = unique_metrics[0]
        else:
            y_label = self.y_col

        # remove per-subplot x-labels
        grid.set_axis_labels("", y_label)

        fig = grid.fig
        if self.title:
            fig.suptitle(self.title, y=0.98)

        # single x-label at the bottom center
        fig.supxlabel("Number of Simulations", x=0.5, y=0.02)

        # Flatten axes list for return
        flat_axes = [ax for sub in axes for ax in sub]
        return fig, flat_axes
