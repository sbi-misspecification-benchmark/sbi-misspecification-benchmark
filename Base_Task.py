from abc import ABC

class BaseTask(ABC):
# return the prior distribution over parameters θ
    def get_prior(self):
        raise NotImplementedError("Task must implement get_prior method")

# return a simulator(function) that maps θ → x
    def simulator(self):
        raise NotImplementedError("Task must implement simulator method")

# return an observation x for a given index
    def get_observation(self, idx: int):
        raise NotImplementedError("Task must implement get_observation method")

