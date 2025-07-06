import random
import torch

from src.evaluation.evaluate_inference import evaluate_inference
from src.inference.Run_Inference import run_inference
from src.tasks.misspecified_tasks import LikelihoodMisspecifiedTask


# Define a dummy task class
class DummyTask2():
    def get_prior(self):
        return torch.distributions.MultivariateNormal(
            loc=torch.zeros(2), covariance_matrix=torch.eye(2)
        )
    def get_reference_posterior_samples(self, idx):
        return torch.ones(100, 2)  # Fake samples

    def get_simulator(self):
        return lambda theta: theta + torch.randn_like(theta)

    def get_observation(self, idx=0):
        return torch.tensor([0.5, 0.5])


# Task registry to hold all available task classes
task_registry = {
    "test_task": DummyTask2,
    "misspecified_likelihood": LikelihoodMisspecifiedTask,
}

def validate_positive(value, default_value):
    """Ensure a configuration value is non-negative, otherwise use the default."""
    if value is None or value < 0:
        return default_value
    return value

def run_benchmark(config):
    random_seed = config.get('random_seed')
    if random_seed is None:
        random_seed = random.randint(0, 2 ** 32 - 1)
        print(f"Generated random seed: {random_seed}")
    else:
        print(f"Using provided random seed: {random_seed}")

    task_name = config.task.name
    if task_name not in task_registry:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(task_registry.keys())}")
    task = task_registry[task_name]()

    method = config.inference.method.upper()
    num_simulations = config.inference.num_simulations
    num_observations = config.inference.num_observations
    num_posterior_samples = config.inference.num_posterior_samples

    print(
        f"\n Running {method} on task {task_name} with {num_simulations} simulations and {num_observations} observations\n")


    run_inference(
        task=task,
        method_name=method,
        num_simulations=num_simulations,
        seed=random_seed,
        num_posterior_samples=num_posterior_samples,
        num_observations=num_observations,
        config=config
    )

    # Evaluation
    metric_name = config.metric.name
    evaluate_inference(
        task=task,
        method_name=method,
        metric_name=metric_name,
        num_observations=num_observations,
        num_simulations=num_simulations,
    )
