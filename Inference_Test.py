import torch
from sbi.inference import NPE
from Run_Inference import run_inference
from Base_Task import BaseTask


class DummyTask(BaseTask):
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
    samples = run_inference(task, method_class= NPE, num_simulations=200)

    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (1000, 2)






