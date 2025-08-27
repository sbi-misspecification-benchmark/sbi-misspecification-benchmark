import os
import torch
from src.evaluation.metrics.c2st import compute_c2st
from src.evaluation.metrics.ppc import compute_ppc

def build_param_folder(task_params):
    """Build param subfolder string from dict, sorted by key."""
    if not task_params:
        return ""
    # Exclude None values for robustness
    items = [(k, v) for k, v in task_params.items() if v is not None]
    items.sort()  # alphabetical order; or change to fixed order if you wish
    return "_".join([f"{k}_{v}" for k, v in items])

def evaluate_inference(task, method_name, metric_name, num_simulations, obs_offset=0, task_params=None):
    """
    Evaluate the metric for exactly one observation (used in loop in benchmark_run.py).

    Args:
        task: Task object with required interface.
        method_name (str): Inference method name.
        metric_name (str): Name of the metric ('c2st', 'ppc', etc).
        num_simulations (int): Simulation count.
        obs_offset (int): Which observation index to evaluate.
        task_params (dict): Dictionary of task parameter names/values for building output path.

    Returns:
        float: metric score for this observation.
    """
    idx = obs_offset
    task_name = task.__class__.__name__
    param_folder = build_param_folder(task_params) if task_params else ""
    if param_folder:
        obs_dir = f'outputs/{task_name}_{method_name}/{param_folder}/sims_{num_simulations}/obs_{idx}'
    else:
        obs_dir = f'outputs/{task_name}_{method_name}/sims_{num_simulations}/obs_{idx}'
    x_path = os.path.join(obs_dir, 'x_obs.pt')
    post_path = os.path.join(obs_dir, 'posterior_samples.pt')

    posterior_samples = torch.load(post_path, map_location='cpu', weights_only=True)

    # assures that the same observation is used
    if os.path.exists(x_path):
        observation = torch.load(x_path, map_location='cpu', weights_only=True)
    else:
        observation = task.get_observation(idx)

    ref_dist = task.get_reference_posterior(observation)
    reference_samples = ref_dist.sample((posterior_samples.shape[0],)).cpu()

    if metric_name == "c2st":
        score = compute_c2st(
            posterior_samples.cpu().numpy(),
            reference_samples.cpu().numpy(),
            test_size=0.3,
            random_state=86,
            plot=True,
            obs_idx=idx+1
        )
    elif metric_name == "ppc":
        simulator = task.get_simulator()
        score = compute_ppc(posterior_samples.cpu(), observation.cpu(), simulator)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    print(f"{metric_name.upper()} for obs {idx}: {score:.3f}")
    return score