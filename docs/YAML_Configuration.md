# YAML Configuration Guide
## Configuration Structure Overview

The benchmark uses a hierarchical configuration system with these main components:

```
configs/
├── main.yaml                 # Primary configuration file
├── task/                     # Task-specific configurations
│   └── misspecified_likelihood.yaml
├── inference/                # Inference method configurations
│   ├── npe.yaml              # Neural Posterior Estimation
│   ├── nle.yaml              # Neural Likelihood Estimation
│   └── nre.yaml              # Neural Ratio Estimation
└── metric/                   # Evaluation metric configurations
    ├── c2st.yaml             # Classifier Two-Sample Test
    └── ppc.yaml              # Posterior Predictive Check

```

## Main Configuration (`main.yaml`)

### Core Parameters

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `defaults` | Specifies which configuration files to inherit from | List | `[task: misspecified_likelihood, inference: npe, metric: c2st]` | 
| `random_seed` | Seed for all random number generators | Integer | 42 |
|`hydra.mode`|Execution mode: `RUN` for single run, `MULTIRUN` for sweeping combinations|String|MULTIRUN|
|`hydra.sweeper.params`|enables multirun with different num_simulations| Dict   | `inference.num_simulations: ${inference.num_simulations}`              |

### Behavior of `hydra.mode: MULTIRUN`

| `inference.num_simulations` value | Behavior                          | Outcome                                  |
|----------------------------------|-----------------------------------|------------------------------------------|
| `100`                            | Single value                      | One run executed                         |
| `100, 1000`                    |  comma-separated values                   | Two runs executed with separate outputs  |
### Note: use MULTIRUN as default, since RUN leads to an error if there are two or more values for num_simulations

Example:
```yaml
defaults:
  - task: misspecified_likelihood
  - inference: npe
  - _self_

random_seed: 42
hydra:
 mode: MULTIRUN
 sweeper:
    params:
      inference.num_simulations: ${inference.num_simulations}
```

## Task Configuration (`task/misspecified_likelihood.yaml`)

### Misspecification Parameters

| Parameter | Description                                                                                                                              | Type | Default |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------|------|---------|
| `task.name` | The name of the inference task to run                                                                                                  | String | `misspecified_likelihood` |
| `task.tau_m` | Controls variance of misspecified component (Float). Higher values >> 1.0 create stronger likelihood misspecification                  | Float | 0.2 | 
| `task.lambda_val` | Mixing weight between well-specified N(θ,I) and misspecified components (Float). Range: 0.0 (normal) to 1.0 (fully misspecified)  | Float | 0.6 | 
| `task.dim` | Dimensionality of parameter space                                                                                                        | Integer | 2 |


Example:
```yaml
task:
  name: misspecified_likelihood
  tau_m: 0.2      # Moderate misspecification
  lambda_val: 0.6  # 60% misspecified component
  dim: 2          # 2D parameter space
```

## Inference Configuration (`inference/*.yaml`)

### Common Parameters

| Parameter | Description | Type | Default       |
|-----------|-------------|------|---------------|
| `inference.method` | Algorithm for simulation-based inference | String | NPE, NLE, NRE |
| `inference.num_simulations` | Number of simulated datasets for training | Integer or comma-separated string | 100-1000  |
| `inference.num_observations` | Number of test observations to evaluate | Integer | 10            |
| `inference.num_posterior_samples` | Samples drawn per posterior distribution  | Integer | 100           |

This benchmark supports the following Simulation-Based Inference (SBI) methods. The choice of method (`inference.method`) determines how the neural network models are trained to estimate the posterior distribution.
* **NPE (Neural Posterior Estimation)**: Directly learns a neural network that approximates the posterior distribution $p(\theta|x)$, which maps observed data $x$ to parameters $\theta$.
* **NLE (Neural Likelihood Estimation)**: Learns a neural network that approximates the likelihood function $p(x|\theta)$, which describes the probability of observing data $x$ given parameters $\theta$. The posterior is then obtained by multiplying with the prior.
* **NRE (Neural Ratio Estimation)**: Learns a classifier that estimates the likelihood-to-evidence ratio $p(x|\theta)/p(x)$ or the posterior-to-prior ratio $p(\theta|x)/p(\theta)$. This ratio can be used for posterior inference.

Example (NPE):
```yaml
inference:
  method: npe
  num_simulations: 100,1000    # More simulations = better accuracy; comparison of accuracy between different values
  num_observations: 10     # Test on 10 different observations
  num_posterior_samples: 100  # More samples = smoother posteriors
```

## Execution Examples
Simply launch the benchmark with the following command
### Usage
```bash
python -m src.run --config-path configs --config-name main
```



