import torch
from src.tasks.misspecified_tasks import GroundTruthModel, LikelihoodMisspecifiedTask


def test_ground_truth_model_shapes():
    model = GroundTruthModel(dim=3)
    samples = model.sample_prior(10)
    assert samples.shape == (10, 3)


def test_likelihood_misspecified_get_true_parameter():
    task = LikelihoodMisspecifiedTask(dim=4, tau_m=0.2, lambda_val=0.6)
    param = task.get_true_parameter(idx=42)
    assert isinstance(param, torch.Tensor)
    assert param.shape == (1, 4)


def test_likelihood_misspecified_get_observation():
    task = LikelihoodMisspecifiedTask(dim=2, tau_m=0.2, lambda_val=0.6)
    obs = task.get_observation(idx=1)
    assert obs.shape == (1, 2)


def test_simulator_output_shape_and_type():
    task = LikelihoodMisspecifiedTask(dim=2, tau_m=0.2, lambda_val=0.6)
    thetas = task.get_true_parameter(0).repeat(100, 1)
    sim_data = task.simulator(thetas)
    assert isinstance(sim_data, torch.Tensor)
    assert sim_data.shape == (100, 2)


def test_reference_posterior_sample_shape():
    task = LikelihoodMisspecifiedTask(dim=2, tau_m=0.2, lambda_val=0.6)
    samples = task.get_reference_posterior_samples(idx=0)
    assert samples.shape == (10_000, 2)


def test_ground_truth_model_accessors():
    model = GroundTruthModel(dim=3)
    assert torch.equal(model.get_mu_prior(), torch.ones(3))
    assert torch.equal(model.get_sigma_prior(), torch.eye(3))
    assert torch.equal(model.get_sigma_likelihood(), torch.eye(3))
    assert model.get_dim() == 3


def test_get_simulator_callable():
    task = LikelihoodMisspecifiedTask(dim=2, tau_m=0.2, lambda_val=0.6)
    simulator_fn = task.get_simulator()
    assert callable(simulator_fn)


def test_simulator_mixes_distributions():
    task = LikelihoodMisspecifiedTask(dim=2,tau_m=0.2, lambda_val=0.6)
    thetas = task.get_true_parameter(0).repeat(1000, 1)
    output = task.simulator(thetas)

    # Since Beta(2,5) samples are typically in [0, 1], and Gaussians aren't,
    # you can heuristically check if both ranges are present
    min_val = output.min().item()
    max_val = output.max().item()

    assert min_val < 0 or max_val > 1  # Simulator likely didn't mix both Beta and Normal samples
