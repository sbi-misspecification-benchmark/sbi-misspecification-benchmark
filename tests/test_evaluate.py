import numpy as np
import torch
import os

from Base_Task import BaseTask
from src.inference.Run_Inference import run_inference
from src.evaluation.metrics.c2st import compute_c2st
from src.evaluation.metrics.ppc import compute_ppc
from src.evaluation.evaluate_inference import evaluate_inference

class DummyTask(BaseTask):
    def __init__(self, dim=2, noise_std=0.5):
        self.dim = dim
        self.noise_std = noise_std
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(dim), torch.eye(dim)
        )

    def get_simulator(self):
        def sim(theta):
            noise = torch.randn_like(theta) * self.noise_std
            return theta + noise  # kein Bias, nur zufälliger noise
        return sim

    def get_reference_posterior_samples(self, idx):
        # Simuliere Verteilung ähnlich wie Posterior mit gleichem noise
        return torch.randn(100, self.dim) * self.noise_std

    def get_observation(self, idx):
        # Observation = theta + noise
        theta = self.prior.sample((1,))
        return theta + torch.randn_like(theta) * self.noise_std

    def get_prior(self):
        return self.prior


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

def test_c2st_similar_distributions():
    """
        Tests if the C2ST returns an accuracy close to 0.5 when both distributions are similar
    """
    np.random.seed(0)

    mean1, mean2 = 0.0, 0.1
    std = 1.0
    n_samples = 1000

    samples_a = np.random.normal(loc=mean1, scale=std, size=(n_samples, 1))
    samples_b = np.random.normal(loc=mean2, scale=std, size=(n_samples, 1))

    score = compute_c2st(samples_a, samples_b, test_size=0.5, random_state=42)


    assert 0.50 < score < 0.75, f"Unexpected C2ST score: {score}"

def test_ppc_high_distance():
    """
    Tests if the PPC distance is high if posterior samples are clearly different

    """
    posterior_samples = torch.randn(100, 2)
    observation = torch.tensor([0.5, 0.5])
    simulator= lambda theta: theta + 2

    score = compute_ppc(posterior_samples, observation, simulator)
    score = float(score)  # Handle tensor return
    print("PPC(high) =", score)
    assert 0.7 <= score <= 1.0  # high distance should yield a high score

def test_ppc_low_distance():
    """Tests if the PPC distance is low for very similar posterior samples"""
    observation = torch.tensor([0.5, 0.5])
    posterior_samples = observation.repeat(100, 1)
    simulator= lambda theta: theta + 0.1   # Simulate small offset

    score = compute_ppc(posterior_samples, observation, simulator)
    score = float(score)  # Handle tensor return
    print("PPC(low) =", score)
    assert 0.0 <= score < 0.3  # Allow a bit of slack for randomness

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