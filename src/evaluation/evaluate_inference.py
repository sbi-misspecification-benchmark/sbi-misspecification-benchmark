import os
import torch
import pandas as pd
from src.evaluation.metrics.c2st import compute_c2st
from src.evaluation.metrics.ppc import compute_ppc


def evaluate_inference(task, method_name, metric_name, num_observations, num_simulations):
    """
    Evaluate how well a chosen inference method estimate the true posterior and save results.

    Args:
        task: an object that implements Base_Task interface and comes with the following
              - the true observation
               - the reference posterior based on the observation
        method_name(str): name of the chosen inference method (e.g. NPE) to load the right data
        metric_name: the name of the metric used for evaluation.py (e.g. c2st)
        num_observations: the number of observations to evaluate (e.g. 1)

    Returns:
        float: the computed score of the metric and saves it as CSV file.
    """
    task_name = task.__class__.__name__
    last_score = 0.0

    for idx in range(num_observations):
        obs_dir = f'outputs/{task_name}_{method_name}/sims_{num_simulations}/obs_{idx}'

        # Load data
        posterior_samples = torch.load(os.path.join(obs_dir, 'posterior_samples.pt'), weights_only=True)
        observation = task.get_observation(idx)
        reference_samples = task.get_reference_posterior_samples(idx)

        # Compute metric
        if metric_name == "c2st":
            score = compute_c2st(
            posterior_samples,
            reference_samples,
            # uses 30% of the data for testing
            test_size=0.3,
            random_state=86
        )

        elif metric_name == "ppc":
            simulator = task.get_simulator()
            score = compute_ppc(posterior_samples, observation, simulator)

        # Save metric in observation directory
        metric_data = {
            "obs_idx": idx,
            "task": task_name,
            "method": method_name,
            "metric": metric_name,
            "score": score,
        }
        pd.DataFrame([metric_data]).to_csv(os.path.join(obs_dir, f"metric_{metric_name}.csv"), index=False)

        last_score = score
        print(f"{metric_name.upper()} for obs {idx}: {score:.3f}")

    return last_score