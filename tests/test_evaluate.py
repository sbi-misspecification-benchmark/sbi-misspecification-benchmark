import torch
import os

from Base_Task import BaseTask
from src.inference.Run_Inference import run_inference
from src.evaluation.metrics.c2st import compute_c2st
from src.evaluation.metrics.ppc import compute_ppc
from src.evaluation.evaluate_inference import evaluate_inference

class DummyTask(BaseTask):
    """
    Dummy task for testing purposes
    """
    def get_simulator(self):
        return lambda theta: theta + 1

    def get_reference_posterior_samples(self, idx):
        return torch.ones(100, 2)  # Fake samples

    def get_observation(self, idx):
        return torch.tensor([0.5, 0.5])

    def get_prior(self):
        return torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

def test_c2st_distinguishes():
    """
    Tests if the C2ST distinguishes between two samples with different distributions

    """
    reference_samples = torch.randn(100, 2)
    inference_samples = torch.randn(100, 2) + 3

    accuracy= compute_c2st(reference_samples, inference_samples, 0.3, 12)

    assert accuracy > 0.9
def test_c2st_on_identical_distribution():
    """
    Tests if the C2ST returns an accuracy close to 0.5 when both distributions are identical
    """
    reference_samples = torch.randn(100, 2)
    inference_samples = torch.randn(100, 2)
    accuracy= compute_c2st(reference_samples, inference_samples, 0.3, 12)
    assert (accuracy-0.5) < 0.1

def test_ppc_high_distance():
    """
    Tests if the PPC distance is high if posterior samples are clearly different

    """
    posterior_samples = torch.randn(100, 2)
    observation = torch.tensor([0.5, 0.5])
    simulator= lambda theta: theta + 2

    distance = compute_ppc(posterior_samples, observation, simulator)
    print(distance)
    assert distance > 1

def test_ppc_low_distance():
    """Tests if the PPC distance is low for very similar posterior samples"""
    observation = torch.tensor([0.5, 0.5])
    posterior_samples = observation.repeat(100, 1)
    simulator= lambda theta: theta + 0.1
    distance = compute_ppc(posterior_samples, observation, simulator)
    print(distance)
    assert distance < 0.2

def test_run_inference_and_evaluate():
    """
    tests whether inference and evaluation are running on one task
    """
    task = DummyTask()
    method = "NPE"
    metric = "c2st"
    seed = 86

    num_simulations = 100
    num_posterior_samples = 50
    num_observations = 1

    # Run inference, samples are saved
    run_inference(
        task,
        method_name=method,
        num_simulations=num_simulations,
        seed=seed,
        num_posterior_samples=num_posterior_samples,
        num_observations=num_observations
    )

    # Evaluates inference, loads saved samples
    score = evaluate_inference(task, method, metric_name=metric, num_observations=num_observations, num_simulations=num_simulations)

    # Checks if score is a float
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0