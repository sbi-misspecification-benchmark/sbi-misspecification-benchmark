import hydra
from omegaconf import DictConfig
import random
import torch
from subprocess import run

from inference.Run_Inference import run_inference
from utils.benchmark_run import validate_positive


# Dummy task registry
class DummyTask():
    def get_prior(self):
        return torch.distributions.MultivariateNormal(
            loc=torch.zeros(2), covariance_matrix=torch.eye(2)
        )
    def get_simulator(self):
        return lambda theta: theta + torch.randn_like(theta)
    def get_observation(self, idx=0):
        return torch.tensor([0.5, 0.5])

task_registry = {
    "test_task": DummyTask
}

@hydra.main(config_path="configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):

    # Get seed or generate it if not specified
    seed = cfg.get("random_seed", None)
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
        print(f"Generated random seed: {seed}")
    else:
        print(f"Using provided random seed: {seed}")

    # Validate task name
    task_name = cfg.task.name
    if task_name not in task_registry:
        raise ValueError(f"Unknown task: '{task_name}'. Available: {list(task_registry.keys())}")
    task = task_registry[task_name]()

    # Parse task config and ensure valid inputs
    num_simulations = validate_positive(cfg.task.num_simulations, 100)
    num_observations = validate_positive(cfg.task.num_observations, 10)
    num_posterior_samples = validate_positive(cfg.task.num_posterior_samples, 50)

    # Convert method input to list
    methods = cfg.methods


    # Run each method
    if cfg.get("multirun", False):
        for method in methods:
            print(f"Launching subprocess for method: {method}")
            run([
                "python", "src/run.py",
                f"+method={method}",
                f"task.name={task_name}",
                f"task.num_simulations={num_simulations}",
                f"task.num_observations={num_observations}",
                f"task.num_posterior_samples={num_posterior_samples}",
                f"random_seed={seed}",
                "multirun=false"
            ])
    else:
        method = methods
        print(f" Running method: {method}")
        run_inference(
            task=task,
            method_name=method,
            num_simulations=num_simulations,
            seed=seed,
            num_posterior_samples=num_posterior_samples,
            num_observations=num_observations,
            config=cfg
        )


if __name__ == "__main__":
    main()
