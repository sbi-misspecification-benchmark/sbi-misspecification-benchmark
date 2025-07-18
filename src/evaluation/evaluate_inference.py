import os
import torch
from src.evaluation.metrics.c2st import compute_c2st
from src.evaluation.metrics.ppc import compute_ppc

def evaluate_inference(task, method_name, metric_name, num_observations, num_simulations, obs_offset=0):
    """
    Evaluate the metric for exactly one observation (used in loop in benchmark_run.py).

    Args:
        task: Task object with required interface.
        method_name (str): Inference method name.
        metric_name (str): Name of the metric ('c2st', 'ppc', etc).
        num_observations (int): Should be 1 here.
        num_simulations (int): Simulation count.
        obs_offset (int): Which observation index to evaluate.

    Returns:
        float: metric score for this observation.
    """
    idx = obs_offset
    task_name = task.__class__.__name__
    obs_dir = f'outputs/{task_name}_{method_name}/sims_{num_simulations}/obs_{idx}'
    posterior_samples = torch.load(os.path.join(obs_dir, 'posterior_samples.pt'), weights_only=True)
    observation = task.get_observation(idx)
    reference_samples = task.get_reference_posterior_samples(idx)

    if metric_name == "c2st":
        score = compute_c2st(
            posterior_samples,
            reference_samples,
            test_size=0.3,
            random_state=86
        )
    elif metric_name == "ppc":
        simulator = task.get_simulator()
        score = compute_ppc(posterior_samples, observation, simulator)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    print(f"{metric_name.upper()} for obs {idx}: {score:.3f}")
    return score