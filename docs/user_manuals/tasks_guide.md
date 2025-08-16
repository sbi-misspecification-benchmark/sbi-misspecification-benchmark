# Adding a New Task

A task defines a prior distribution, a simulator (the generative model), how to fetch an observation, and optionally how to provide reference posterior samples for evaluation.


## Implementation

### 1. Set up a Python file
### 1.1 Required Methods
All tasks must implement these methods:

- *get_prior(self)*   
returns the prior distribution 

- *simulator(self, thetas: torch.Tensor)*   
simulates observations x given parameters `thetas` under specific model.

- *get_observation(self, idx: int)*
returns a observation for a given index

- *get_reference_posterior_samples(idx)*   
provides ground-truth posterior samples if available.




### 1.2 Example 
For better understanding on how to implemnt the required methods, take a look at the following example `src/tasks/my_task.py` or for a more complex implemention look at `src\tasks\misspecified_tasks.py` or `src\tasks\linear_gaussian_task.py`. 

Example:    
This simple example uses a normal prior and simulates data by adding Gaussian noise.

<pre>import torch
import torch.distributions as D

class MyTask():
    def get_prior(self):
        return D.Normal(0, 1)   # use normal prior
    def simulator(self, theta):
        return theta + torch.randn_like(theta)
    def get_observation(self, idx):
        return torch.tensor([0.0])
    def get_reference_posterior_samples(self, idx):
        return torch.randn(100, 1)</pre>





### 2. Add a new Config File
For it to be recognised by Hydra you then need to add a new .yaml config file `src/configs/task/my_task.yaml`, that follows the simple structure:

Example:    
<pre>name: my_task
mu: 2.0
sigma: 1.0</pre> 

Note: Config keys are task-specific. Each task can define its own parameters (e.g., `tau_m`, `lambda_val` for the misspecified task, or `prior_mu`, `prior_sigma` for the provided example with Gaussian noise). Nevertheless the keys must match the arguments expected by the task's `__init__` method.

Now you can call the new task in the main config file `main.yaml`.   

Example:
<pre> defaults:
  - task: my_task     # Call new task here
  - inference: npe
  - metric: c2st
  - _self_</pre> 


### 3. Update the Registry
For the runner to find and initaite the new task that is stated in the config, you have to add the new task to the task registry in `src\utils\benchmark_run.py`. The resulting dictionary should have the following stucture: 

<pre>task_registry = {
    "misspecified_likelihood": LikelihoodMisspecifiedTask,
    "my_task": MyTask,
}</pre> 

## Expected Behavior
- The runner will recognize the new task via Hydra.
- Posterior samples and metrics will be saved under *outputs/*.