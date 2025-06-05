import argparse
import yaml
import random
import os
import csv
from src.inference.Run_Inference import run_inference
from src.metrics.basic_metrics import c2st_metric, ppc_metric

# ... DummyTask, task_registry, load_config, validate_positive ...

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks using a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    random_seed = config.get('random_seed', random.randint(0, 2**32 - 1))

    task = config.get('task', {})
    task_name = task.get('name', "Default Task Name")
    if task_name not in task_registry:
        raise ValueError(f"Task '{task_name}' not found.")

    inference = config.get('inference', {})
    num_simulations = validate_positive(inference.get('num_simulations'), 100)
    num_observations = validate_positive(inference.get('num_observations'), 10)
    num_posterior_samples = validate_positive(inference.get('num_posterior_samples'), 50)
    method = inference.get('method', "npe")
    metrics_list = config.get('metrics', ["c2st", "ppc"])

    task_class = task_registry[task_name]
    task_instance = task_class()

    # Get output dir and ensure it exists
    output_base = f"outputs/{task_name}_{method}/sim_{num_simulations}"
    os.makedirs(output_base, exist_ok=True)
    csv_path = os.path.join(output_base, "metrics.csv")
    csv_header = ["task", "method", "num_simulations", "obs_idx"] + metrics_list
    metrics_rows = []

    # Run inference and evaluate for each observation
    for obs_idx in range(num_observations):
        samples = run_inference(
            task=task_instance,
            method_name=method,
            num_simulations=num_simulations,
            seed=random_seed,
            num_posterior_samples=num_posterior_samples,
            num_observations=1,  # Only one at a time
            config=config
        )
        # Compute metrics
        metric_values = []
        if "c2st" in metrics_list:
            metric_values.append(c2st_metric(samples, ground_truth="dummy"))
        if "ppc" in metrics_list:
            metric_values.append(ppc_metric(samples, observation=task_instance.get_observation(obs_idx)))
        # ... add additional metrics if needed ...

        # Write row
        row = [task_name, method, num_simulations, obs_idx] + metric_values
        metrics_rows.append(row)

    # Save metrics CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for row in metrics_rows:
            writer.writerow(row)

    print(f"Saved metrics csv: {csv_path}")

if __name__ == "__main__":
    main()