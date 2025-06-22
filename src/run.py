import hydra
from omegaconf import DictConfig
import random
import torch


from src.inference.Run_Inference import run_inference
from src.utils.benchmark_run import validate_positive

"""
This script loads a configuration using Hydra and performs inference
runs for all methods specified in the given task


Expected config parameters (from main.yaml):
- task:
    - name: name of the task (must be in task_registry)
    - num_simulations: number of simulations per run
    - num_observations: number of observations per run
    - num_posterior_samples: number of posterior samples
- methods: list of inference methods to run
- random_seed: (optional) random seed for reproducibility

This is the version not using Hydras multirun
"""

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

    # Run each method
    method = cfg.inference.method.upper()
    num_simulations = validate_positive(cfg.inference.num_simulations, 100)
    num_observations = validate_positive(cfg.inference.num_observations, 10)
    num_posterior_samples = validate_positive(cfg.inference.num_posterior_samples, 50)

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