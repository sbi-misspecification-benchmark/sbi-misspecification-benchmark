import random
import torch
import os
import pandas as pd
from omegaconf import OmegaConf

from src.evaluation.evaluate_inference import evaluate_inference
from src.inference.Run_Inference import run_inference
from src.tasks.misspecified_tasks import LikelihoodMisspecifiedTask
from src.utils.save_results import save_results

# Task registry to hold all available task classes
task_registry = {
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

    Task = task_registry[task_name]
    task_kwargs = OmegaConf.to_container(config.task, resolve=True) or {}
    task_kwargs.pop("name", None)
    task = Task(**task_kwargs)

    method = config.inference.method.upper()
    num_simulations = config.inference.num_simulations
    num_observations = config.inference.num_observations
    num_posterior_samples = config.inference.num_posterior_samples

    # the observation is fixed here and passed during the benchmarking process
    observations = [task.get_observation(i) for i in range(num_observations)]


    print(
        f"\n Running {method} on task {task_name} with {num_simulations} simulations and {num_observations} observations\n")

    run_inference(
        task=task,
        method_name=method,
        num_simulations=num_simulations,
        seed=random_seed,
        num_posterior_samples=num_posterior_samples,
        num_observations=num_observations,
        config=config,
        observations=observations,
    )

    # Prepare task parameter columns for results (all except name, dim)
    task_params = {
        k: v for k, v in OmegaConf.to_container(config.task, resolve=True).items()
        if k not in ["name", "dim"]
    }

    # Determine which metrics to compute based on config
    metric_config = config.metric.name
    compute_c2st = metric_config in ["c2st", "c2st_ppc"]
    compute_ppc = metric_config in ["ppc", "c2st_ppc"]

    # Evaluation: collect all metrics for all obs, save one metrics.csv
    all_metrics = []
    for obs_idx in range(num_observations):

        x_o = observations[obs_idx]
        metrics_dict = {"obs_idx": obs_idx, "task": task_name, "method": method}


        if compute_c2st:
            c2st_score = evaluate_inference(
                task=task,
                method_name=method,
                metric_name="c2st",
                num_simulations=num_simulations,
                obs_offset=obs_idx,
                task_params=task_params  # pass param info for correct path
            )
            all_metrics.append({
                "metric": "c2st",
                "value": c2st_score,
                "task": task_name,
                "method": method,
                "num_simulations": num_simulations,
                "observation_idx": obs_idx,
                **task_params
            })

        if compute_ppc:
            ppc_score = evaluate_inference(
                task=task,
                method_name=method,
                metric_name="ppc",
                num_simulations=num_simulations,
                obs_offset=obs_idx,
                task_params=task_params  # pass param info for correct path
            )
            all_metrics.append({
                "metric": "ppc",
                "value": ppc_score,
                "task": task_name,
                "method": method,
                "num_simulations": num_simulations,
                "observation_idx": obs_idx,
                **task_params
            })

    # Save all results as rows in metrics.csv
    if all_metrics:
        import pandas as pd
        param_folder = "_".join([f"{k}_{task_params[k]}" for k in sorted(task_params)]) if task_params else ""
        output_dir = f"outputs/{task_name}_{method}"
        if param_folder:
            output_dir += f"/{param_folder}"
        output_dir += f"/sims_{num_simulations}"
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    print(f"Saved metrics for all observations and parameters.")