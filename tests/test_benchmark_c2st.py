import torch

from src.inference.Run_Inference import run_inference
from src.evaluation.metrics.c2st import compute_c2st
from src.tasks.misspecified_tasks import LikelihoodMisspecifiedTask

# Fix random seed
seed = 42
torch.manual_seed(seed)
def test_benchmark_low_misspecification():
    """
        Tests if the C2ST returns an accuracy close to 0.5 when misspecification is low
    """
# Instantiate task with low misspecification
    task = LikelihoodMisspecifiedTask(tau_m=1.1, lambda_val=0.01, dim=2)

# Choose observation index
    idx = 0

# Get observation
    x_o = task.get_observation(idx)

# Get reference posterior distribution
    posterior_dist = task.get_reference_posterior(x_o)

# Draw samples from both distributions
    reference_samples = posterior_dist.sample((1000,))
    inference_samples = run_inference(task, "NPE", 1000, 1000, 1,None, observations=[x_o])

# Compute C2ST
    accuracy = compute_c2st(
        inference_samples.numpy(),
        reference_samples.numpy(),
        test_size=0.3,
        random_state=seed)
    assert (accuracy-0.5) < 0.05

def test_benchmark_high_misspecification():
    """
        Tests if the C2ST returns an accuracy significantly greater than 0.5 when misspecification is high
    """
# Instantiate task with high misspecification
    task = LikelihoodMisspecifiedTask(tau_m=10, lambda_val=0.99, dim=2)

    idx = 0

    x_o = task.get_observation(idx)

    posterior_dist = task.get_reference_posterior(x_o)

    reference_samples = posterior_dist.sample((1000,))
    inference_samples = run_inference(task, "NPE", 1000, 1000, 1,None, observations=[x_o] )

    accuracy = compute_c2st(
        inference_samples.numpy(),
        reference_samples.numpy(),
        test_size=0.3,
        random_state=seed)
    assert (accuracy-0.5) > 0.15


def test_benchmark_only_reference():
    """
    Tests if the C2ST returns an accuracy close to 0.5 when both distributions are similar and there is no misspecification
    """
    task = LikelihoodMisspecifiedTask(tau_m=20, lambda_val=0.99, dim=2)

    idx = 0

    x_o = task.get_observation(idx)

    posterior_dist = task.get_reference_posterior(x_o)

    reference_samples = posterior_dist.sample((1000,))
    inference_samples = posterior_dist.sample((1000,))

    accuracy = compute_c2st(
        inference_samples.numpy(),
        reference_samples.numpy(),
        test_size=0.3,
        random_state=seed)
    assert (accuracy-0.5) < 0.05

