# This script didn't run for me if I didnt set the KMP_DUPLICATE_LIB_OK environment variable
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from sbi.inference import NPE
from src.inference.Run_Inference import run_inference
import os
#from Base_Task import BaseTask




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

    def get_observation(self, idx= 0):
        return torch.tensor([0.5, 0.5])


def test_run_inference():
    task = DummyTask()
    
    
    samples = run_inference(task, 
                            method_name = "NPE", 
                            num_simulations = config["task"]["num_simulations"], 
                            num_posterior_samples = config["task"]["num_posterior_samples"], 
                            num_observations = config["task"]["num_observations"],
                            seed= 42,
                            config = config)

    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (20, 2)

    # Check if new folders are created for each observation 
    for idx in range(config["task"]["num_observations"]):
        outdir = f"outputs/DummyTask_NPE/obs_{idx}"
        assert os.path.exists(outdir), f"Missing output dir: {outdir}"
        assert os.path.exists(os.path.join(outdir, "posterior_samples.pt"))
        assert os.path.exists(os.path.join(outdir, "config_used.yaml"))


