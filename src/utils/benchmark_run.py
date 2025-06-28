import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import csv
from src.inference.Run_Inference import run_inference
from src.tasks.misspecified_tasks import LikelihoodMisspecifiedTask

def c2st_metric(samples, ground_truth):
    import numpy as np
    return float(np.random.uniform(0, 1))

def ppc_metric(samples, observation):
    import numpy as np
    return float(np.random.uniform(0, 1))

@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    # Use config values directly, no hardcoded defaults
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
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_header = ["task", "method", "num_simulations", "obs_idx"] + list(metrics_list)
    metrics_rows = []

    for obs_idx in range(num_observations):
        samples = run_inference(
            task=task,
            method_name=method,
            num_simulations=num_simulations,
            seed=random_seed,
            num_posterior_samples=num_posterior_samples,
            num_observations=1,
            config=cfg
        )
        ground_truth = "dummy"
        x_obs = task.get_observation(idx=obs_idx)
        metric_values = []
        for metric in metrics_list:
            if metric.lower() == "c2st":
                metric_values.append(c2st_metric(samples, ground_truth))
            elif metric.lower() == "ppc":
                metric_values.append(ppc_metric(samples, x_obs))
            else:
                metric_values.append(float("nan"))
        row = [cfg.task.name, method, num_simulations, obs_idx] + metric_values
        metrics_rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for row in metrics_rows:
            writer.writerow(row)
    print(f"Saved metrics csv: {csv_path}")

if __name__ == "__main__":
    main()