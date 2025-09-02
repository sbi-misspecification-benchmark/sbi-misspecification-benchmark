from abc import ABC, abstractmethod
from torch.distributions import Distribution 
from torch import Tensor

class BaseTask(ABC):
    """Base class for all SBI Tasks."""

    @abstractmethod
    def get_prior(self) -> Distribution:
        """Return the prior distribution p(theta) over parameters theta."""
        raise NotImplementedError("Task must implement get_prior method")
    

    @abstractmethod
    def get_true_parameter(self, idx: int) -> Tensor:
        """Return the deterministic 'true' parameter for index `idx`."""
        raise NotImplementedError("Task must implement get_true_parameter method")
    
    @abstractmethod
    def get_observation(self, idx: int) -> Tensor:
        """Return an observation x for a given index 'idx'"""
        raise NotImplementedError("Task must implement get_observation method")
    
    @abstractmethod
    def get_reference_posterior(self, observation: Tensor) -> Distribution:
        """Return the reference posterior p(theta | observation)."""
        raise NotImplementedError("Task must implement get_reference_posterior method")
    
    @abstractmethod
    def get_reference_posterior_samples(self, idx: int) -> Tensor:
        """Return the samples for a given index 'idx' from a reference posterior p(theta | observation)"""
        raise NotImplementedError("Task must implement get_reference_posterior_samples")

    @abstractmethod
    def simulator(self, theta) -> Tensor:
        """Simulate observations x given theta"""
        raise NotImplementedError("Task must implement simulator method")

    def get_simulator(self):
        """Return a simulator function."""
        return self.simulator
    
    
    


    

    
