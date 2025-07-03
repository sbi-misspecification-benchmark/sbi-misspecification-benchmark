import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from src.inference.Run_Inference import run_full_benchmark
from src.tasks.misspecified_tasks import LikelihoodMisspecifiedTask

@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    task = LikelihoodMisspecifiedTask(
        tau_m=cfg.task.tau_m,
        lambda_val=cfg.task.lambda_val
    )
    method = cfg.inference.method.upper()
    num_simulations = cfg.inference.num_simulations
    num_observations = cfg.inference.num_observations
    num_posterior_samples = cfg.inference.num_posterior_samples
    metrics_list = cfg.metrics.metrics
    random_seed = cfg.get("random_seed", None)
    if random_seed is not None:
        torch.manual_seed(random_seed)
    output_dir = f"outputs/{cfg.task.name}_{method}/sim_{num_simulations}"

    run_full_benchmark(
        task=task,
        method_name=method,
        num_simulations=num_simulations,
        num_observations=num_observations,
        num_posterior_samples=num_posterior_samples,
        metrics_list=metrics_list,
        output_dir=output_dir,
        seed=random_seed,
        cfg=cfg,
    )

if __name__ == "__main__":
    main()