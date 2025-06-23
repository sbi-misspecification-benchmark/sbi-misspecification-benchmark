import argparse
import yaml
import random
import os
import csv

from src.inference.Run_Inference import run_inference
from src.tasks.misspecified_tasks import LikelihoodMisspecifiedTask

# ---- Example metric functions ----
# Replace these with your real implementations!
def c2st_metric(samples, ground_truth):
    # Placeholder: random value for demonstration
    import numpy as np
    return float(np.random.uniform(0, 1))

def ppc_metric(samples, observation):
    # Placeholder: random value for demonstration
    import numpy as np
    return float(np.random.uniform(0, 1))

# Define a dummy task class
class DummyTask:
    def simulator(self, thetas):
        print(f"Running DummyTask simulator with thetas: {thetas}")

    def get_observation(self, idx=0):
        # Example: return a dummy observation
        import torch
        return torch.tensor([0.5, 0.5])

# Task registry to hold all available task classes
task_registry = {
    "test_task": DummyTask,
    "misspecified_likelihood": LikelihoodMisspecifiedTask,
}

def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def validate_positive(value, default_value):
    """Ensure a configuration value is non-negative, otherwise use the default."""
    if value is None or value < 0:
        return default_value
    return value

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks using a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Handle random seed
    random_seed = config.get('random_seed')
    if random_seed is None:
        random_seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {random_seed}")
    else:
        print(f"Using provided random seed: {random_seed}")

    # Process task
    task_cfg = config.get('task', {})
    task_name = task_cfg.get('name', "Default Task Name")

    # Check if the task exists in the registry
    if task_name not in task_registry:
        raise ValueError(
            f"Task '{task_name}' not found in the registry. "
            f"Available tasks are: {list(task_registry.keys())}"
        )

    # Read inference config (prefer top-level inference block for modern configs)
    inference_cfg = config.get('inference', {})
    num_simulations = validate_positive(inference_cfg.get('num_simulations'), 100)
    num_observations = validate_positive(inference_cfg.get('num_observations'), 10)
    num_posterior_samples = validate_positive(inference_cfg.get('num_posterior_samples'), 50)
    method = inference_cfg.get('method', "npe")

    # Read metrics config (list of metric names)
    metrics_list = config.get('metrics', ["c2st", "ppc"])
    # If metrics is a dict (from Hydra), use metrics['metrics']
    if isinstance(metrics_list, dict) and "metrics" in metrics_list:
        metrics_list = metrics_list["metrics"]

    print(
        f"Executing task: {task_name}\n"
        f"num_simulations: {num_simulations}\n"
        f"num_observations: {num_observations}\n"
        f"num_posterior_samples: {num_posterior_samples}\n"
        f"method: {method}\n"
        f"metrics: {metrics_list}"
    )

    # Instantiate the task
    task_class = task_registry[task_name]
    task_instance = task_class()

    # Prepare output dir
    output_dir = f"outputs/{task_name}_{method}/sim_{num_simulations}"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_header = ["task", "method", "num_simulations", "obs_idx"] + metrics_list
    metrics_rows = []

    # Loop over observations, run inference and evaluation for each
    for obs_idx in range(num_observations):
        # Run inference for single observation
        samples = run_inference(
            task=task_instance,
            method_name=method,
            num_simulations=num_simulations,
            seed=random_seed,
            num_posterior_samples=num_posterior_samples,
            num_observations=1,  # only for this observation
            config=config
        )

        # Prepare ground truth/obs for metrics
        # For real tasks, replace with actual ground truth and observations
        ground_truth = "dummy"  # replace as needed
        x_obs = task_instance.get_observation(idx=obs_idx)

        # Calculate metrics
        metric_values = []
        for metric in metrics_list:
            if metric.lower() == "c2st":
                metric_values.append(c2st_metric(samples, ground_truth))
            elif metric.lower() == "ppc":
                metric_values.append(ppc_metric(samples, x_obs))
            else:
                metric_values.append(float("nan"))  # or raise error

        # Record row
        row = [task_name, method, num_simulations, obs_idx] + metric_values
        metrics_rows.append(row)

    # Write metrics.csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for row in metrics_rows:
            writer.writerow(row)

    print(f"Saved metrics csv: {csv_path}")

if __name__ == "__main__":
    main()