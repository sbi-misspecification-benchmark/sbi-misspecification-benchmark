# Understanding the Code:

## Class: GroundTruthModel:
- Simulated model for a baysian inference problem 
- Prior ("knowledge"/assumtion about parameters beforehand) 
- Likelihood (model generated from those parameters)  

### 1. Initializing
- Initializing the mean, prior parameters and the likelihood

### 2. Sampling
- Based on prior parameters and the likelihood values will be generated to have more training data.

### 3. Get reference posterior
- The Bayesian posterior can be computed analytically.
- Returns the posterior mean and covariance for a given set of observations.




## Class: LikelihoodMisspecifiedTask()
- simulates a Bayesian inference task by using a misspecified Likelihood-Modell
- generates data, that does not fully match the model assumed during inference 

### 1. Initializing
- Initializing the parameter dimension, misspecified likelihood variance and the mixture weight (lambda) 
- setting up a GroundTruthModel that simulates the real world


### 2. Get true parameter
- get specific parameter for given index that is used to simulate a synthetic observation.

### 3. Get observation
- generate a "true" observation by using the true parameters

### 4. Get reference posterier samples
- calculate the posterior from Groundtruth


