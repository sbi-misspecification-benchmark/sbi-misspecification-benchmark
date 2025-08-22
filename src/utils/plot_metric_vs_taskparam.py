import argparse
from pathlib import Path

from src.utils.LinePlot import LinePlot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot evaluation metrics as a function of task parameters."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the metrics_all.csv file."
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Name of the metric to plot (e.g. c2st, ppc)."
    )
    parser.add_argument(
        "--task_param",
        type=str,
        help="Task parameter column to use for the x-axis (e.g. tau_m, lambda_val)."
    )
    return parser.parse_args()


def plot_metric_vs_taskparam(input_path: str, metric: str, task_param: str) -> None:
    """
    Generate and save a line plot of a chosen evaluation metric
    against a given task parameter from a metrics CSV file.

    The function:
      - Loads benchmark results from `input_path` (metrics_all.csv format).
      - Filters the data for the specified `metric`.
      - Creates a line plot with `task_param` on the x-axis and the metric values on the y-axis.
      - Saves the plot to `outputs/{task}_{method}/plots/{metric}_vs_{task_param}.png`,
        where `{task}` and `{method}` are inferred from the data.

    Args:
        input_path (str): Path to the metrics CSV file (e.g. "metrics_all.csv").
        metric (str): Name of the metric to plot (case-sensitive, e.g. "C2ST").
        task_param (str): Column name of the task parameter to use for the x-axis
                          (e.g. "tau_m", "lambda_val").

    Returns:
        None: The plot is saved to disk.
    """

    # Initialize the LinePlot Instance
    plotter = LinePlot(data_sources=input_path, x=task_param, log_x=False, filename=f"{metric}__vs__{task_param}.png")

    # Load the data
    df = plotter.load_data()

    # Extract rows that match the metric
    df_filtered = df[df["metric"] == metric]

    # Create the plot and get the Figure
    fig, _ = plotter.plot(df_filtered)

    # Build save dir and filename from the filtered data
    task = df_filtered["task"].unique().item()
    method = df_filtered["method"].unique().item()

    plotter.save_directory = Path(f"outputs/{task}_{method}/plots").resolve()


    # Save the figure
    plotter.save(fig)



def main():
    args = parse_args()
    plot_metric_vs_taskparam(
        input_path=args.input_path,
        metric=args.metric,
        task_param=args.task_param,
    )


if __name__ == "__main__":
    main()
