import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
from .inference.Run_Inference import run_inference
from .tasks.misspecified_tasks import LikelihoodMisspecifiedTask

@hydra.main(config_path="../src/configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    # Initialize the task model based on config
    task = LikelihoodMisspecifiedTask(
        tau_m=cfg.task.task.tau_m,
        lambda_val=cfg.task.task.lambda_val
    )
    cfg.inference.inference.method = cfg.inference.inference.method.upper()

    if cfg.random_seed is not None:
        torch.manual_seed(cfg.random_seed)

    # Create the output directory
    output_dir = Path(
        f"outputs/"
        f"{cfg.task.task.name}_"
        f"{cfg.inference.inference.method.upper()}/"
        f"sims_{cfg.inference.inference.num_simulations}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through each observation and generate posterior samples
    for obs_idx in range(cfg.inference.inference.num_observations):
        obs_dir = output_dir / f"obs_{obs_idx}"
        obs_dir.mkdir(exist_ok=True)

        x_o = task.get_observation(obs_idx)
        torch.save(x_o, obs_dir / "observation.pt")

        samples = run_inference(
            task=task,
            method_name=cfg.inference.inference.method,
            num_simulations=cfg.inference.inference.num_simulations,
            seed=cfg.random_seed
        )
        torch.save(samples, obs_dir / "posterior_samples.pt")


if __name__ == "__main__":
    main()