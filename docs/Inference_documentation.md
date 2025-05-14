# Understanding the structure of `run_inference` and DummyTask

---

## Function: `run_inference(task, method_name, num_simulations)`

This function performs simulation-based inference (SBI) given:

* A task (provides prior, simulator, observation)
* A method name (NPE, NLE, or NRE)
* A number of simulations

---

### Goal

To generate samples from the **posterior distribution** $p(\theta \mid x_0)$, using simulation-based inference techniques.
The function is modular and supports multiple inference algorithms.

---

### Key Steps

| Step                      | Description                                                                     |
| ------------------------- | ------------------------------------------------------------------------------- |
| 1. Get Prior              | Calls `task.get_prior()` to access the prior distribution over parameters       |
| 2. Get Simulator          | Uses `task.get_simulator()` to define how to generate `x` from `θ`              |
| 3. Simulate Training Data | Draws `(θ, x)` pairs by sampling from prior and running the simulator |
| 4. Train Inference        | Uses `sbi` inference method (e.g. NPE) to learn a posterior approximation       |
| 5. Posterior Sampling     | Uses the trained posterior to generate new samples given an observation         |

---

## Class: `DummyTask`

A mock task used for testing.
Simulates a very basic Bayesian model without misspecification.

---

### Key Components

| Method              | Description                                                                |
| ------------------- | -------------------------------------------------------------------------- |
| `get_prior()`       | Returns a standard Normal distribution $N(0, I)$ in 2D                     |
| `get_simulator()`   | Defines a function `x = θ + ε` where ε \~ N(0, I)                          |
| `get_observation()` | Returns a fixed observation (e.g. `[0.5, 0.5]`) to condition the posterior |

---


### Example Usage

```python
def test_run_inference():
    task = DummyTask()
    print("Im finished")
    for method_name in ["NPE", "NLE", "NRE"]:
        samples = run_inference(task, method_name=method_name, num_simulations=200)

assert isinstance(samples, torch.Tensor)
        assert samples.shape == (1000, 2)
```

---

### What It Tests

* That `run_inference()` works for all methods ("NPE", "NLE", "NRE")
* That the posterior has the expected shape and type



You can use any task that implements:

* `get_prior()`
* `get_simulator()`
* `get_observation(idx)`

and use this function to run inference.
