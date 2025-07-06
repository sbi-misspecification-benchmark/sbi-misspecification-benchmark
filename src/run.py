import hydra
from omegaconf import DictConfig
from src.utils.benchmark_run import run_benchmark

@hydra.main(config_path="configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    """Execute benchmark with the given Hydra configuration.

        Args:
            cfg: Hydra configuration object containing:
                - task: Task configuration (name, parameters)
                - inference: Method and simulation parameters
                - metric: Evaluation metric(s) to compute
                - random_seed: Random seed for reproducibility
        """
    run_benchmark(cfg)

if __name__ == "__main__":
    main()