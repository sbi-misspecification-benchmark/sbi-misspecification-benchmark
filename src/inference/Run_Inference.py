# This script didn't run for me if I didnt set the KMP_DUPLICATE_LIB_OK environment variable
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from omegaconf import OmegaConf
import torch
from sbi.inference import NPE, NLE, NRE
import yaml
import os

# List of inference methods
methods = {
    "NPE": NPE,
    "NLE": NLE,
    "NRE": NRE,
}


def run_inference(task, method_name, num_simulations, seed=None, num_posterior_samples = 50, num_observations = 10, config = None):

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

    # Loop over observations
    for idx in range(num_observations):
        x_obs = task.get_observation(idx=idx)
        samples = posterior.sample((num_posterior_samples,), x=x_obs)

        # Create a new folder for each observation and save results
        output_dir = f"outputs/{task_name}_{method_name}/sims_{num_simulations}/obs_{idx}/"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(samples, os.path.join(output_dir, "posterior_samples.pt"))

        # Save config in each observation folder
        if config is not None:
            with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
                yaml.dump(OmegaConf.to_container(config, resolve=True), f)

    return samples