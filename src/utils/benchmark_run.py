import random
import torch
import os
import pandas as pd

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

    # Evaluation: collect all metrics for all obs, save one metrics.csv
    metric_name = config.metric.name
    all_metrics = []
    for obs_idx in range(num_observations):
        score = evaluate_inference(
            task=task,
            method_name=method,
            metric_name=metric_name,
            num_observations=1,
            num_simulations=num_simulations,
            obs_offset=obs_idx
        )
        # evaluate_inference returns a dict per obs
        all_metrics.append({
            "obs_idx": obs_idx,
            "task": task_name,
            "method": method,
            "metric": metric_name,
            "score": score
        })
    # Save metrics.csv in sims_{num_simulations} dir
    outdir = f"outputs/{task_name}_{method}/sims_{num_simulations}"
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(os.path.join(outdir, "metrics.csv"), index=False)
    print(f"Saved metrics for all observations to {os.path.join(outdir, 'metrics.csv')}")