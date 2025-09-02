from pathlib import Path
import pandas as pd

from hydra.experimental.callback import Callback
from omegaconf import OmegaConf

from src.utils.LinePlot import LinePlot
from src.utils.consolidate_metrics import consolidate_metrics
from src.utils.benchmark_run import task_registry


class PostProcessCallback(Callback):
    def on_multirun_end(self, config, **kwargs):
        # 1) Gather run directories
        sweep_dir = Path(config.hydra.sweep.dir)                    # Sweep directory of the multirun
        job_dirs = [d for d in sweep_dir.iterdir() if d.is_dir()]   # Job directories of all the single runs


        # 2) Collect run information (config parameters + path to benchmark results) into a DataFrame
        run_records = []    # Each record will hold task, method, num_simulations and the path to its metrics.csv

        for job_dir in job_dirs:
            # Load the config of this job/run
            cfg_path = job_dir / ".hydra" / "config.yaml"
            cfg = OmegaConf.load(cfg_path)

            # Extract relevant config parameters (task, method, num_simulations)
            task_name = config.task.name
            if task_name not in task_registry:
                raise ValueError(f"Unknown task: {task_name}. Available: {list(task_registry.keys())}")

            # Initialize with arbitrary params to only infer the task class name
            task_class_name = task_registry[task_name].__name__

            method = str(cfg.inference.method).upper()
            num_simulations = int(cfg.inference.num_simulations)

            # Derive path to benchmark results file metrics.csv
            metrics_path = Path("outputs") / f"{task_class_name}_{method}" / f"sims_{num_simulations}" / "metrics.csv"

            # Append record
            run_records.append({
                "task": task_class_name,
                "method": method,
                "num_simulations": num_simulations,
                "metrics_path": metrics_path,
            })

        df = pd.DataFrame(run_records)

        


        # 3) Visualize
        # 3.1) Get the data sources
        metrics_paths = df["metrics_path"].tolist()

        # 3.2) Get the save directory
        # Get unique task-method pairs
        unique_task_methods = df[["task", "method"]].drop_duplicates()

        # Consolidate all metrics.csv files into one DataFrame
        for _, row in unique_task_methods.iterrows():
            task = row["task"]
            method = row["method"].upper()
            base_dir = Path("outputs") / f"{task}_{method}"

            for sim_dir in base_dir.glob("sims_*"):
                per_metric_files = list(sim_dir.glob("metrics_*.csv"))
                if not per_metric_files:
                    continue

                if len(per_metric_files) == 1:
                    df = pd.read_csv(per_metric_files[0])
                else:
                    df = pd.concat((pd.read_csv(p) for p in per_metric_files), ignore_index=True)

                df.to_csv(sim_dir / "metrics.csv", index=False)


        if len(unique_task_methods) == 1:
            # Only one unique task-method combination
            task = unique_task_methods.iloc[0]["task"]
            method = unique_task_methods.iloc[0]["method"].upper()

            save_directory = Path(f"outputs/{task}_{method}/plots")
        else:
            # Multiple task-method combinations
            save_directory = Path("outputs/plots")

        # 3.3) Consolidate metrics.csv files to metrics_all.csv files within their respective task_method folder
        for _, row in unique_task_methods.iterrows():
            task = row["task"]
            method = row["method"].upper()
            input_dir = Path("outputs") / f"{task}_{method}"
            output_file = input_dir / "metrics_all.csv"

            consolidate_metrics(input_dir=input_dir, output_file=output_file)


        # 3.3) Create and Save the Plot
        plotter = LinePlot(data_sources=metrics_paths, save_directory=save_directory)
        plotter.run()
