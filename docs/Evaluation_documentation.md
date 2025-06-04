
---

## Function: `evaluate_inference(task, method_name, metric_name)`

This function evaluates how well an inference method (e.g. NPE) approximates the true posterior using metrics C2ST or PPC. 

Preconditions are:

Posterior samples have been generated and saved using run_inference

The task provides a ground-truth posterior for comparison

---

### Key Steps

| Step              | Description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| 1. Load Data      | Loads posterior, observation, and reference samples                        |
| 2. Compute Metric | Uses `compute_c2st()` or `compute_ppc()` based on input                    |
| 3. Save Results   | Stores the result in CSV format in the corresponding task/method directory |

---

## Available Metrics

| Metric | Description                                                                                    |
| ------ | ---------------------------------------------------------------------------------------------- |
| `c2st` | Trains a classifier to distinguish inference generated posterior samples from true posterior samples                       |
| `ppc`  | Compares how well simulated observations using posterior samples match the real one            |

---


## Example Usage

```python

task = DummyTask()
evaluate_inference(task=task, method_name="NPE", metric_name="c2st")
```

Expected output:

```bash
C2ST for DummyTask/NPE: 0.850
```

---

## Dependencies

* The `task` must implement:

  * `get_simulator()`
  * `get_reference_posterior_samples(idx)`
  * `get_observation(idx)`

---

