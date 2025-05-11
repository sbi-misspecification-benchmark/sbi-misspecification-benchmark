import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from sbi.inference import NPE, NLE, NRE

# List of inference methods
methods = {
    "NPE": NPE,
    "NLE": NLE,
    "NRE": NRE,
}

def run_inference(task, method_name, num_simulations):
    if method_name not in methods:
        raise ValueError(f"Method {method_name} is not supported. Choose from {list(methods.keys())}.")
    
    method_class = methods[method_name]
    
    prior = task.get_prior()
    simulator = task.get_simulator()
# draw parameters from prior
    θ = prior.sample((num_simulations,))

# simulate data
    x = simulator(θ)

# create and train inference model
    inference = method_class(prior)
    density_estimator = inference.append_simulations(θ, x).train()

# perform inference
    posterior = inference.build_posterior(density_estimator)
    x_o = task.get_observation(idx=0)
    samples = posterior.sample((1000,), x=x_o)

    return samples