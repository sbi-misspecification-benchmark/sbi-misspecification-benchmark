import torch
from pathlib import Path
from src.evaluation.metrics.c2st import compute_c2st
from src.evaluation.metrics.ppc import compute_ppc

def evaluate_inference(task, method_name, metric_name, num_simulations, obs_offset=0):
    """
    Evaluate the metric for exactly one observation (used in loop in benchmark_run.py).

    Args:
        task: Task object with required interface.
        method_name (str): Inference method name.
        metric_name (str): Name of the metric ('c2st', 'ppc', etc).
        num_simulations (int): Simulation count.
        obs_offset (int): Which observation index to evaluate.

    Returns:
        float: metric score for this observation.
    """
    idx = obs_offset
    task_name = task.__class__.__name__
    obs_dir = Path(f"outputs/{task_name}_{method_name}/sims_{num_simulations}/obs_{idx}")
    x_path = obs_dir/"x_obs.pt"
    post_path = obs_dir/"posterior_samples.pt"

    # Load posterior_samples.pt; raise if missing
    if post_path.exists():  
        posterior_samples = torch.load(post_path, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f"Missing posterior samples at {post_path}.")


    # Load x_obs.pt; raise if missing
    if x_path.exists():
        observation = torch.load(x_path, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f"Missing observations at {x_path}.")


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
