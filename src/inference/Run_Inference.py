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

def run_inference(
    task,
    method_name,
    num_simulations,
    seed=None,
    num_posterior_samples=50,
    x_obs=None,
    trained_density_estimator=None,
    posterior=None,
    config=None,
    obs_idx=None,
    train_only=False
):
    """Train model if train_only, or sample conditioned on x_obs if not."""
    if method_name not in methods:
        raise ValueError(f"Method {method_name} is not supported. Choose from {list(methods.keys())}.")

    if seed is not None:
        torch.manual_seed(seed)

    if train_only:
        # TRAINING PHASE (run only once)
        print("Training model...")
        prior = task.get_prior()
        simulator = task.get_simulator()
        theta = prior.sample((num_simulations,))
        x = simulator(theta)
        print(f"Simulated theta shape: {theta.shape}, x shape: {x.shape}")
        inference = methods[method_name](prior)
        density_estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(density_estimator)
        print(f"Posterior built: {posterior}")
        return density_estimator, posterior

    else:
        # INFERENCE PHASE (one observation at a time)
        if posterior is None:
            raise ValueError("Posterior must be provided for inference.")
        if x_obs is None:
            raise ValueError("x_obs must be provided for inference.")

        samples = posterior.sample((num_posterior_samples,), x=x_obs)

        # Save results
        task_name = task.__class__.__name__
        output_dir = f"outputs/{task_name}_{method_name}/obs_{obs_idx}/"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(samples, os.path.join(output_dir, "posterior_samples.pt"))

        # Save config in each observation folder
        if config is not None:
            with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
                yaml.dump(config, f)

        return samples