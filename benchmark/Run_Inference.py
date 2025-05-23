# This script didn't run for me if I didnt set the KMP_DUPLICATE_LIB_OK environment variable
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from sbi.inference import NPE, NLE, NRE

# List of inference methods
methods = {
    "NPE": NPE,
    "NLE": NLE,
    "NRE": NRE,
}

def run_inference(task, method_name, num_simulations, seed = None):
    """Run simulation-based inference on a given task using the specified method.
    Returns samples from the posterior conditioned on the true observation"""
    if method_name not in methods:
        raise ValueError(f"Method {method_name} is not supported. Choose from {list(methods.keys())}.")
    
    method_class = methods[method_name]
    
    prior = task.get_prior()
    simulator = task.get_simulator()

    if seed is not None:
        torch.manual_seed(seed)
 
# draw parameters from prior
    theta = prior.sample((num_simulations,))

# simulate data
    x = simulator(theta)

# create and train inference model
    inference = method_class(prior)
    density_estimator = inference.append_simulations(theta, x).train()

# perform inference
    posterior = inference.build_posterior(density_estimator)
    x_o = task.get_observation(idx=0)
    samples = posterior.sample((1000,), x=x_o)

    return samples
