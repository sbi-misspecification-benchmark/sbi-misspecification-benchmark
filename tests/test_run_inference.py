import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.inference.Run_Inference import run_inference

config = {
    "method": "NPE",
    "task": {
        "name": "DummyTask",
        "num_simulations": 10,
        "num_observations": 2,
        "num_posterior_samples": 20
    },
    "random_seed": 123
}

class DummyTask():
    def get_prior(self):
        return torch.distributions.MultivariateNormal(
            loc=torch.zeros(2), covariance_matrix=torch.eye(2)
        )
    def get_simulator(self):
        return lambda theta: theta + torch.randn_like(theta)
    def get_observation(self, idx=0):
        return torch.tensor([0.5, 0.5])

def test_run_inference():
    task = DummyTask()
    num_obs = config["task"]["num_observations"]
    num_samples = config["task"]["num_posterior_samples"]
    num_sims = config["task"]["num_simulations"]
    method = config["method"]

    # Train only once
    trained_density_estimator, posterior = run_inference(
        task=task,
        method_name=method,
        num_simulations=num_sims,
        seed=42,
        train_only=True
    )

    # Loop over observations, as in the runner
    for idx in range(num_obs):
        x_obs = task.get_observation(idx)
        samples = run_inference(
            task=task,
            method_name=method,
            num_simulations=num_sims,
            num_posterior_samples=num_samples,
            x_obs=x_obs,
            trained_density_estimator=trained_density_estimator,
            posterior=posterior,
            config=config,
            obs_idx=idx,
            train_only=False
        )
        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (num_samples, 2)

        outdir = f"outputs/DummyTask_NPE/obs_{idx}"
        assert os.path.exists(outdir), f"Missing output dir: {outdir}"
        assert os.path.exists(os.path.join(outdir, "posterior_samples.pt"))
        assert os.path.exists(os.path.join(outdir, "config_used.yaml"))