import torch
import torch.distributions as dist


class LinearGaussianTask:
    """
    Implements a linear Gaussian model for SBI with the relationship: x = Aθ + ε, where:
    - θ ~ N(prior_mean, prior_cov) are the parameters
    - ε ~ N(0, noise_cov) is observation noise
    - x is the observed data

    It provides analytical posterior computation for testing SBI methods.
    Key features include:
    - Configurable dimension (dim)
    - Adjustable prior (prior_mean, prior_cov)
    - Controllable noise (noise_cov)
    - Reproducible observations (using seed and caching)
    """
    def __init__(self, dim=2, seed=42):
        """
        Initialize the linear Gaussian model.

        Args:
            dim (int): Dimension of parameter space θ (default: 2)
            seed (int): Random seed for reproducibility (default: 42)
        """
        self.dim = dim
        self.seed = seed

        # Model parameters
        self.prior_mean = torch.zeros(dim)
        self.prior_cov = torch.eye(dim)
        self.A = torch.eye(dim)
        self.noise_cov = 0.1 * torch.eye(dim)

        # Caching for observations and posteriors
        self.saved_observations = {}
        self.saved_posteriors = {}

        # Set seed for reproducibility
        torch.manual_seed(seed)

    def get_prior(self):
        """
        Get the prior distribution over parameters θ.

        Returns:
        dist.MultivariateNormal: The prior distribution p(θ) = N(prior_mean, prior_cov)
        """
        return dist.MultivariateNormal(self.prior_mean, self.prior_cov)

    def simulator(self, theta):
        """
        Simulate observations x given parameters θ.
        Implements x = Aθ + ε where ε ~ N(0, noise_cov)

        Args:
            theta (torch.Tensor): Parameter values with shape (batch_size, dim)

        Returns:
            torch.Tensor: Generated observations with shape (batch_size, dim)
        """
        theta = theta.view(-1, self.dim)
        noise = dist.MultivariateNormal(torch.zeros(self.dim), self.noise_cov).sample(theta.shape[:1])
        return torch.mm(theta, self.A.t()) + noise

    def get_simulator(self):
        """Returns the simulator function."""
        return self.simulator

    def get_observation(self, idx):
        """Returns a reproducible observation for a given index."""
        if idx not in self.saved_observations:
            theta = self.get_prior().sample()
            x = self.simulator(theta.unsqueeze(0))[0]
            self.saved_observations[idx] = x
        return self.saved_observations[idx]

    def _compute_posterior(self, x):
        """
        Compute the analytical posterior p(θ|x).

        Args:
            x (torch.Tensor): Observation data with shape

        Returns:
            dist.MultivariateNormal: Posterior distribution p(θ|x) = N(post_mean, post_cov)
        """
        A_T = self.A.t()
        noise_cov_inv = torch.inverse(self.noise_cov)
        prior_cov_inv = torch.inverse(self.prior_cov)

        post_cov = torch.inverse(A_T @ noise_cov_inv @ self.A + prior_cov_inv)
        post_mean = post_cov @ (A_T @ noise_cov_inv @ x + prior_cov_inv @ self.prior_mean)

        return dist.MultivariateNormal(post_mean, post_cov)

    def get_reference_posterior_samples(self, idx, num_samples=10000):
        """
        Generates samples from the analytical posterior p(θ|x) for given observation index.

        Args:
            idx (int): Observation index
            num_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Posterior samples with shape (num_samples, dim)
        """
        if idx not in self.saved_posteriors:
            x = self.get_observation(idx)
            posterior = self._compute_posterior(x)
            self.saved_posteriors[idx] = posterior

        return self.saved_posteriors[idx].sample((num_samples,))


if __name__ == "__main__":
    """
       Showcase of LinearGaussianTask features.
       Shows basic usage and checks essential operations.
    """

    # Initialize with default parameters
    task = LinearGaussianTask()

    # Show prior sampling
    prior = task.get_prior()
    print("Prior sample:", prior.sample())

    # Demonstrate simulator
    theta_test = torch.tensor([[1.0, 2.0]])
    x_sim = task.simulator(theta_test)
    print("Simulated x:", x_sim)

    # Get reproducible observation
    x_obs = task.get_observation(0)
    print("Observation:", x_obs)

    # Show posterior samples
    post_samples = task.get_reference_posterior_samples(0, num_samples=3)
    print("Posterior samples:")
    print(post_samples)