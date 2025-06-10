import torch
from src.tasks.linear_gaussian_task import LinearGaussianTask

def test_prior_sample_shapes():
    # Checks the shape of samples from the prior distribution
    task = LinearGaussianTask(dim=3)
    prior = task.get_prior()
    samples = prior.sample((10,))
    assert samples.shape == (10, 3)


def test_simulator_output_shape():
    # Checks the shape of simulated data for a set of parameters
    task = LinearGaussianTask(dim=2)
    thetas = torch.randn(100, 2)
    sim_data = task.simulator(thetas)
    assert sim_data.shape == (100, 2)


def test_get_observation_shape():
    # Checks the shape of one single observation
    task = LinearGaussianTask(dim=4)
    obs = task.get_observation(idx=1)
    assert obs.shape == (4,)


def test_reference_posterior_sample_shape():
    # Checks the shape of samples from the posterior distribution
    task = LinearGaussianTask(dim=2)
    samples = task.get_reference_posterior_samples(idx=0, num_samples=10000)
    assert samples.shape == (10000, 2)


def test_task_properties():
    # Confirms that the properties of the task are properly initialized
    task = LinearGaussianTask(dim=3)
    assert task.dim == 3
    assert torch.equal(task.A, torch.eye(3))
    assert torch.equal(task.noise_cov, 0.1 * torch.eye(3))


def test_get_simulator_callable():
    # Makes sure that the simulator function can be found and used
    task = LinearGaussianTask()
    simulator_fn = task.get_simulator()
    assert callable(simulator_fn)


def test_observation_reproducibility():
    # Checks that the observations can be repeated using the same seed and index
    task = LinearGaussianTask(seed=42)
    obs1 = task.get_observation(idx=1)
    obs2 = task.get_observation(idx=1)
    assert torch.equal(obs1, obs2)


def test_posterior_changes_with_observation():
    # Checks that different observations lead to different posteriors
    task = LinearGaussianTask(seed=42)

    obs0 = task.get_observation(0)
    samples0 = task.get_reference_posterior_samples(0)

    obs1 = task.get_observation(1)
    samples1 = task.get_reference_posterior_samples(1)

    assert not torch.allclose(obs0, obs1)
    assert not torch.allclose(samples0.mean(0), samples1.mean(0))