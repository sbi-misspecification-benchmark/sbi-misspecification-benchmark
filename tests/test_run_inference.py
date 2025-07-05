import torch
from src.inference.Run_Inference import run_inference

class DummyTask():
    def get_prior(self):
        return torch.distributions.MultivariateNormal(
            loc=torch.zeros(2), covariance_matrix=torch.eye(2)
        )
    def get_simulator(self):
        return lambda theta: theta + torch.randn_like(theta)

    def get_observation(self, idx=0):
        return torch.tensor([0.5, 0.5])

    def get_reference_posterior_samples(self, idx):
        return torch.ones(100, 2)  # Fake samples

def test_run_inference():
    task = DummyTask()
    num_simulations = 200
    num_posterior_samples = 50
    num_observations = 10
    for method_name in ["NPE", "NLE", "NRE"]:
        samples = run_inference(
            task,
            method_name=method_name,
            num_simulations=num_simulations,
            num_posterior_samples=num_posterior_samples,
            num_observations=num_observations
        )

        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (num_posterior_samples, 2)