from omegaconf import OmegaConf
import torch
from sbi.inference import NPE, NLE, NRE
import yaml
from pathlib import Path


# List of inference methods
methods = {
    "NPE": NPE,
    "NLE": NLE,
    "NRE": NRE,
}


def run_inference(
    task,
    method_name,
    num_simulations,
    num_posterior_samples,
    num_observations,
    seed=None,
    config=None,
    observations=None,
):

    """
    Run simulation-based inference on a given task using the specified method.

    Args:
        task: The task object providing prior, simulator, and observation interface.
        method_name: The inference method to use ("NPE", "NLE", or "NRE").
        num_simulations: Number of simulations for training.
        seed: Random seed for reproducibility.
        num_posterior_samples: Number of posterior samples to generate per observation.
        num_observations: Number of observations to loop over.
        config: (optional) Configuration dictionary.
            NOTE: This parameter is used **solely for saving the configuration** to disk (e.g., as 'config_used.yaml').
            It is **not** used for extracting values such as `task`, `method_name`, or `num_observations`.
            All required arguments must be passed explicitly.
        observations: (optional) Observations passed by benchmark_run.py to loop over.

    Returns:
        samples: Posterior samples from the last observation.
    """
    if method_name not in methods:
        raise ValueError(f"Method {method_name} is not supported. Choose from {list(methods.keys())}.")

    method_class = methods[method_name]

    prior = task.get_prior()
    simulator = task.get_simulator()

    if seed is not None:
        torch.manual_seed(seed)

    # draw parameters from prior
    theta = prior.sample((num_simulations,))

    # simulate data
    x = simulator(theta)

    # create and train inference model
    inference = method_class(prior)
    density_estimator = inference.append_simulations(theta, x).train()

    # perform inference
    posterior = inference.build_posterior(density_estimator)


    task_name = task.__class__.__name__  # get task name

# for using Run_inference.py independently
    if observations is None:
        observations = [task.get_observation(i) for i in range(num_observations)]

    # Loop over observations
    for idx in range(num_observations):
        x_obs = observations[idx]
        # shape handling
        if x_obs.ndim == 2 and x_obs.shape[0] == 1:
            x_obs = x_obs.squeeze(0)

        samples = posterior.sample((num_posterior_samples,), x=x_obs)

        # Create a new folder for each observation and save results
        output_dir = Path("outputs") / f"{task_name}_{method_name}" / f"sims_{num_simulations}" / f"obs_{idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(samples, output_dir / "posterior_samples.pt")
        torch.save(x_obs, output_dir / "x_obs.pt") # saves the observation to be used for evaluation

        # Save config in each observation folder
        if config is not None:
            config_path = output_dir / "config_used.yaml"
            with config_path.open("w") as f:
                yaml.dump(OmegaConf.to_container(config, resolve=True), f)

    return samples