from abc import ABC

class BaseTask(ABC):
    """Base class for SBI task that defines the required interface."""

    def get_prior(self):
        """return the prior distribution over parameters θ"""
        raise NotImplementedError("Task must implement get_prior method")

    def simulator(self, theta):
        """Simulate x given theta"""
        raise NotImplementedError("Task must implement simulator method")


    def get_simulator(self):
        """return a simulator(function)"""
        return self.simulator


    def get_observation(self, idx: int):
        """return an observation x for a given index"""
        raise NotImplementedError("Task must implement get_observation method")

    def get_reference_posterior_samples(self, idx: int):
        """return the "true" posterior samples for a given index"""
        raise NotImplementedError("Task must implement get_reference_posterior_samples")
