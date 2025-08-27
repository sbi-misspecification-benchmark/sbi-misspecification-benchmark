import torch
import torch.distributions as dist
from src.tasks.Base_Task import BaseTask


class GroundTruthModel:
    """Ground truth model with standard Gaussian prior and identity covariance likelihood."""

    def __init__(self, dim=2):
        """Initialize mean, prior and likelihood"""
        self.dim = dim
        self.mu_prior = torch.ones(dim)  # Prior mean is set to 1
        self.sigma_prior = torch.eye(dim)  # Prior covariance is identity
        self.sigma_likelihood = torch.eye(dim)  # Likelihood covariance is identity matrix

        # Create distributions
        self.prior_dist = dist.MultivariateNormal(self.mu_prior, self.sigma_prior)


    def get_mu_prior(self):
        """Return the prior mean vector."""
        return self.mu_prior


    def get_sigma_prior(self):
        """Return the prior covariance matrix."""
        return self.sigma_prior


    def get_sigma_likelihood(self):
        """Return the likelihood covariance matrix."""
        return self.sigma_likelihood


    def get_dim(self):
        """Return the dimensionality of the distribution."""
        return self.dim



    def sample_prior(self, num_samples=1):
        """Sample from N(mu_prior, sigma_prior)."""
        return self.prior_dist.sample((num_samples,))


    def sample_data(self, parameters):
        """Generate one observation from N(parameters, sigma_likelihood) for each parameter vector."""
        parameters = parameters.view(-1, self.dim)

        # Generate one sample for each parameter vector
        data = torch.stack([
            dist.MultivariateNormal(param, self.sigma_likelihood).sample()
            for param in parameters
        ])

        # Shape will be (num_parameters, dim)
        return data


    def get_reference_posterior(self, observations):
        """Get the reference posterior for a given set of observations."""

        mean = 0.5 * (observations + self.mu_prior)
        cov = self.sigma_likelihood/2
        return dist.MultivariateNormal(mean, cov)





class LikelihoodMisspecifiedTask(BaseTask):
    """Task for inference in a Gaussian model with misspecified likelihood."""


    def __init__(
        self,
        dim: int,
        tau_m: float,  # Misspecified likelihood variance
        lambda_val: float,  # Mixture weight
    ):
        """Initialize the Gaussian misspecified likelihood task.

        Args:
            dim (int): Dimensionality of the parameter space
            tau_m (float): Variance factor for the misspecified likelihood
            lambda_val (float): Mixture weight in [0, 1]
        """
        self.dim = int(dim)

        # misspeciifcation parameters
        self.tau_m = float(tau_m)
        self.lambda_val = float(lambda_val)

        # prior parameters
        self.mu_prior = torch.ones(self.dim)
        self.sigma_prior = torch.eye(self.dim)

        # Setup ground truth model
        self.ground_truth = GroundTruthModel(dim=self.dim)

        # define prior
        self.prior = self.ground_truth.prior_dist


    def get_prior(self):
        """Return the prior distribution."""
        return self.prior


    def get_true_parameter(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Get the true parameter for a given index.

        Args:
            idx (int): Index of the parameter
            device (str): Device to use

        Returns:
            torch.Tensor: True parameter
        """
        torch.manual_seed(idx)
        return self.prior.sample((1,)).to(device)


    def get_observation(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Get the observation for a given index.

        Args:
            idx (int): Index of the observation
            device (str): Device to use

        Returns:
            torch.Tensor: Observation
        """
        theta_o = self.get_true_parameter(idx, device)
        return self.ground_truth.sample_data(theta_o)


    def get_reference_posterior_samples(self, idx: int, device: str = "cpu"):
        """Get samples from the reference posterior.

        Args:
            idx (int): Index of the observation
            device (str): Device to use

        Returns:
            torch.Tensor: Posterior samples
        """
        torch.manual_seed(idx)
        x_o = self.get_observation(idx, device)
        samples = self.ground_truth.get_reference_posterior(x_o).sample((10_000,))
        return samples.reshape(10_000, self.dim)


    def simulator(self, thetas:torch.Tensor):
        """Simulate observations x given parameters theta under a misspecified likelihood model.

        Args:
            thetas: - of shape (batch_size, dim)
                    - containing params (vectors) from which observations are simulated

        Returns:
            torch.Tensor: - of shape (batch_size, dim)
                          - containing simulated observations
                          - each observation corresponds to an input param
                          - result models likelihood misspecification by combining samples from different
                            distributions
        """
        # thetas shape: (batch_size, dim)
        batch_size = thetas.shape[0]

        # Generate a batch of Bernoulli samples - decide which distribution to use for each sample
        is_beta = torch.bernoulli(torch.tensor(self.lambda_val, dtype=torch.float32).expand(batch_size))

        # Initialize result tensor with the same shape as thetas
        result = torch.zeros_like(thetas)

        # Process beta samples
        beta_mask = is_beta == 1
        beta_samples = dist.Beta(torch.tensor(2.), torch.tensor(5.)).sample((beta_mask.sum(), self.dim))
        result[beta_mask] = beta_samples

        # Process normal samples
        # For samples where is_beta=0, we sample from N(theta, tau_m * I)
        normal_mask = ~beta_mask
        normal_dist = dist.MultivariateNormal(
            loc=thetas[normal_mask],
            covariance_matrix=self.tau_m * torch.eye(self.dim)
        )
        result[normal_mask] = normal_dist.sample()

        return result


    def get_simulator(self):
        """Return simulator function."""
        return self.simulator

    def get_reference_posterior(self, observation: torch.Tensor):
        """Return the reference posterior distribution given an observation."""
        if observation.ndim == 2:
            observation = observation.squeeze(0)

        mean = 0.5 * (observation + self.mu_prior)
        cov = self.ground_truth.get_sigma_likelihood() / 2

        return dist.MultivariateNormal(mean, cov)


if __name__ == "__main__":
    num_samples = 1000

    # just some dummy testing
    task = LikelihoodMisspecifiedTask()
    thetas = task.get_true_parameter(0).repeat(num_samples, 1)
    print("True parameters are:", task.get_true_parameter(0))
    print("Observations are:", task.simulator(thetas))