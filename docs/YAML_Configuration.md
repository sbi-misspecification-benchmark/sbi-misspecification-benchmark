# YAML Configuration 

This documentation explains how to configure and use the config.yaml files for running a benchmark.

## 🔧 Configuration Fields   

These are the necessary fields that need to be configured in the `config.yaml` with a short explanation, the expected type and possible default values.

| Field                 | Description                                                              | Expected Type              | Default value        |
|----------------------|---------------------------------------------------------------------------|------------------------------------|--------------------------|
| `task.name`          | The name of the inference task to run.                                    | String                             | No default          |
| `num_simulations`    | Number of training simulations to generate.                               | Integer                            | No default               |
| `num_observations`   | Number of observations to generate posterior inferences for.              | Integer                            | No default               |
| `num_posterior_samples` | Number of posterior samples per observation.                           | Integer                            | No default               |
| `random_seed`        | Seed used for reproducibility.                                            | Integer                            | Will be generated randomly                       |
| `method`             | Name of the inference method to use.                                      | String                             | No default              |

## 🧪 Example Configuration

Here is an example for a `config.yaml` configuration file:

```yaml
task:
  name: Likelihood_misspecifictaion_task    
  num_simulations: 10000    
  num_observations: 25
  num_posterior_samples: 1000
random_seed: 42
method: NLE