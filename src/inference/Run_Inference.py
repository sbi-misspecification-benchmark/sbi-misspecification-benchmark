from omegaconf import OmegaConf
import torch
from sbi.inference import NPE, NLE, NRE
import yaml
import os

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
    if method_name not in methods:
        raise ValueError(f"Method {method_name} is not supported. Choose from {list(methods.keys())}.")

    method_class = methods[method_name]
    prior = task.get_prior()
    simulator = task.get_simulator()

    if seed is not None:
        torch.manual_seed(seed)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    inference = method_class(prior)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)

    task_name = task.__class__.__name__

    if observations is None:
        observations = [task.get_observation(i) for i in range(num_observations)]

    # Get task parameters for folder structure
    known_attrs = {"dim", "mu_prior", "sigma_prior", "ground_truth", "prior"}
    task_params = {
        k: v for k, v in vars(task).items()
        if k not in known_attrs and not k.startswith('_')
    }
    # Build folder string, sorted for consistency
    param_folder = "_".join([f"{k}_{task_params[k]}" for k in sorted(task_params)])

    for idx in range(num_observations):
        x_obs = observations[idx]
        if x_obs.ndim == 2 and x_obs.shape[0] == 1:
            x_obs = x_obs.squeeze(0)
        samples = posterior.sample((num_posterior_samples,), x=x_obs)

        output_dir = f"outputs/{task_name}_{method_name}"
        if param_folder:
            output_dir += f"/{param_folder}"
        output_dir += f"/sims_{num_simulations}/obs_{idx}/"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(samples, os.path.join(output_dir, "posterior_samples.pt"))
        torch.save(x_obs, os.path.join(output_dir, "x_obs.pt"))
        if config is not None:
            with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
                yaml.dump(OmegaConf.to_container(config, resolve=True), f)
    return samples