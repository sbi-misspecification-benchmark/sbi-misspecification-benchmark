import os
import torch
import pandas as pd
from src.utils.io_utils import load_tensor, save_file
from src.evaluation.metrics.c2st import compute_c2st
from src.evaluation.metrics.ppc import compute_ppc


def evaluate_inference(task, method_name, metric_name, num_observations ):
    """
    Evaluate how well a chosen inference method estimate the true posterior

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
    scores=[]
    for idx in range (num_observations):
        base_path = f'outputs/{task_name}_{method_name}/obs_{idx}'
        posterior_samples = torch.load(os.path.join(base_path, 'posterior_samples.pt'))
        observation = task.get_observation(idx)
        reference_samples = task.get_reference_posterior_samples(idx)

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
        scores.append({
            "obs_idx": idx,
            "task": task_name,
            "method": method_name,
            "metric": metric_name,
            "score": score,
        })


    df = pd.DataFrame([{
        "task": task_name,
        "method": method_name,
        "metric": metric_name,
        "score": score
    }])
    save_file(task_name, method_name, "evaluation.csv", df)


    print(f"{metric_name.upper()} for {task_name}/{method_name}: {score:.3f}")
    return score



