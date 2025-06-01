import torch
import torch.distributions as dist


class LinearGaussianTask:
    """Linear Gaussian Task (x = Aθ + ɛ) without misspecification."""
    def __init__(self, dim=2, seed=42):
        """Initializes model parameters and caching for reproducibility."""
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
        """Returns the prior distribution over theta."""
        return dist.MultivariateNormal(self.prior_mean, self.prior_cov)

    def simulator(self, theta):
        """Simulates observation x from theta: x = A*theta + noise."""
        theta = theta.view(-1, self.dim)
        noise = dist.MultivariateNormal(torch.zeros(self.dim), self.noise_cov).sample(theta.shape[:1])
        return torch.mm(theta, self.A.t()) + noise

    def get_simulator(self):
        """Returns the simulator function."""
        return self.simulator

    def get_observation(self, idx):
        """Retrieves a reproducible observation for a given index."""
        if idx not in self.saved_observations:
            theta = self.get_prior().sample()
            x = self.simulator(theta.unsqueeze(0))[0]
            self.saved_observations[idx] = x
        return self.saved_observations[idx]

    def _compute_posterior(self, x):
        """Computes the analytical posterior distribution P(theta|x)"""
        A_T = self.A.t()
        noise_cov_inv = torch.inverse(self.noise_cov)
        prior_cov_inv = torch.inverse(self.prior_cov)

        post_cov = torch.inverse(A_T @ noise_cov_inv @ self.A + prior_cov_inv)
        post_mean = post_cov @ (A_T @ noise_cov_inv @ x + prior_cov_inv @ self.prior_mean)

        return dist.MultivariateNormal(post_mean, post_cov)

    def get_reference_posterior_samples(self, idx, num_samples=10000):
        """Samples from the analytical posterior for a given observation index."""
        if idx not in self.saved_posteriors:
            x = self.get_observation(idx)
            posterior = self._compute_posterior(x)
            self.saved_posteriors[idx] = posterior

        return self.saved_posteriors[idx].sample((num_samples,))


if __name__ == "__main__":
    # Simple test
    task = LinearGaussianTask()

    prior = task.get_prior()
    print("Prior sample:", prior.sample())

    theta_test = torch.tensor([[1.0, 2.0]])
    x_sim = task.simulator(theta_test)
    print("Simulated x:", x_sim)

    x_obs = task.get_observation(0)
    print("Observation:", x_obs)

    post_samples = task.get_reference_posterior_samples(0, num_samples=3)
    print("Posterior samples:")
    print(post_samples)