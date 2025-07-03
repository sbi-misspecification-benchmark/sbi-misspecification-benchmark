import os
import csv
from src.evaluation.evaluate_inference import compute_c2st, compute_ppc

def run_full_benchmark(
    task,
    method_name,
    num_simulations,
    num_observations,
    num_posterior_samples,
    metrics_list,
    output_dir,
    seed,
    cfg,
):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_header = ["task", "method", "num_simulations", "obs_idx"] + list(metrics_list)
    metrics_rows = []

    for obs_idx in range(num_observations):
        samples = run_inference(
            task=task,
            method_name=method_name,
            num_simulations=num_simulations,
            seed=seed,
            num_posterior_samples=num_posterior_samples,
            num_observations=1,
            config=cfg,
        )
        ground_truth = task.get_ground_truth(idx=obs_idx)
        x_obs = task.get_observation(idx=obs_idx)
        metric_values = []
        for metric in metrics_list:
            if metric.lower() == "c2st":
                metric_values.append(compute_c2st(samples, ground_truth, threshold=0.3, seed=12))
            elif metric.lower() == "ppc":
                metric_values.append(compute_ppc(samples, x_obs))
            else:
                metric_values.append(float("nan"))
        row = [task.name, method_name, num_simulations, obs_idx] + metric_values
        metrics_rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for row in metrics_rows:
            writer.writerow(row)
    print(f"Saved metrics csv: {csv_path}")