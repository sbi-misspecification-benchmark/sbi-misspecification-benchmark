import hydra
from omegaconf import DictConfig
from src.utils.benchmark_run import run_benchmark


@hydra.main(config_path="configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    """Hydra entrypoint: runs the benchmark for one config."""
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
