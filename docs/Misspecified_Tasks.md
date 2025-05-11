# Understanding the structure of a messpecified task

## Class: GroundTruthModel:
Represents a simple Bayesian model where the posterior can be calculated analytically.

- **Prior**: "knowledge"/assumtion about parameters beforehand 
- **Likelihood**: model generated from those parameters 
- **Posterior**: Calculated as the average of the prior mean and the observed data

### Key Methods
- '__init__': Initializing the mean, prior parameters and the likelihood    
- 'sample_prior(n)': sample
- 'sample_data(theta)': Based on prior parameters and the likelihood values will be generated to have training data.
- 'get_reference_posterior(x)': Returns the posterior distribution (posterior mean and covarince for a given set of observation)   



## Class: LikelihoodMisspecifiedTask()
Simulates a Bayesian inference task where the data is generated from a misspecified likelihood that does not fully match the assumed model.    

Uses "GroundTruthModel to simulate the "true world". Data is generated from a mixture of:
    - a normal distribution (approximate match) 
    - a beta distribution (clearly wrong) 


### Key Methods
- '__init__': Initialize the parameters in the model, a misspecified likelihood variance and a mixture weight.
-'get_true_parameter(idx)': get specific parameter for given index that is used to simulate a synthetic observation.
- 'get_observation(idx)': generate a "true" observation by using the true parameters
- 'get_reference_posterior_samples(idx)': sample from the true posterior
- 'simulator(thetas)'simulate observation x under a misspecified likelihoodmodel


## Example

```python
num_samples = 1000
task = LikelihoodMisspecifiedTask()
thetas = task.get_true_parameter(0).repeat(num_samples, 1)
print("True parameters are:", task.get_true_parameter(0))
print("Observations are:", task.simulator(thetas))