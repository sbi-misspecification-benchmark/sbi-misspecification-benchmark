from pathlib import Path

from hydra.experimental.callback import Callback
from omegaconf import OmegaConf

from src.utils.LinePlot import LinePlot


class PostProcessCallback(Callback):
    def on_multirun_end(self, config, **kwargs):

        # 1) Get the file directories to the metrics.csv files that were created in the current multirun sweep
        sweep_dir = Path(config.hydra.sweep.dir)               # Get the Sweeping directory of the current run
        job_dirs = [d for d in sweep_dir.iterdir() if d.is_dir()]   # ...

        metrics_paths = []

        for job_dir in job_dirs:
            # Load the config
            cfg_path = job_dir / ".hydra" / "config.yaml"
            cfg = OmegaConf.load(cfg_path)

            # Extract ...
            task = str(cfg.task.name)
            method = str(cfg.inference.method)
            num_sims = int(cfg.inference.num_simulations)

            p = Path("outputs") / f"{task}_{method}" / f"sims_{num_sims}" / "metrics.csv"   # relative to project root
            metrics_paths.append(p)

        # 2) Visualize
        plotter = LinePlot(metrics_paths)
        plotter.run()







        # Visualization
        # TODO Call Visualization Plotter with the List of file directories
        # TODO Extract the consolidated file from the plotter.
        # TODO QUESTION: where should we safe the respective df of the current plot.
        #  What would the name be? (As of now only think for multiple simulations not tasks or methods)

        # Consolidation of task_method directories
        # TODO where do we get the information from on how to consolidate these directories?
        # TODO Actually right now its easy, just consolidate task_method directory.
        #  Extract this information from the loaded config file.
