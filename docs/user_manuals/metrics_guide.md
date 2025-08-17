# Adding a New Metric

A metric evaluates how close inferred posteriors are to ground-truth reference distributions.

## Implementation and Example
### 🐍 1. Set up a Python file
If you want to add a new metric, create a new python file `src/evaluation/metrics/mymetric.py`.


### ✅ 1.1 Required Methods
Since metrics can differ significantly in their logic (e.g. C2ST trains a classifier, while PPC generates data and compares it to observations), there is no shared interface that all metrics must implement.       
However, each metric should implement the main function named `compute_mymetric(...){...}` that takes the necessary inputs (e.g. posterior samples, reference data) and returns a scalar score.



### 💡 1.2 Example 
For a better understanding of the main function, take a look at the following short example. This simple metric measures the absolute difference in means between posterior and reference distributions.   

*Example: compute_mymetric*
```python
def compute_mymetric(posterior_samples, reference_samples):
    return float(abs(posterior_samples.mean() - reference_samples.mean()))
```




### 📜 2. Add a new Config File
For the metric to be recognised by Hydra you then need to add a new .yaml config file `src/configs/metric/mymetric.yaml`, that follows the simple structure:

*Example: Config File*    
```yaml
name: mymetric
```

Now you can call the new metric in the main config file `main.yaml`.   

*Example: Main.yaml Configuration*
```yaml
defaults:
  - task: misspecified_likelihood     
  - inference: npe
  - metric: mymetric       # Call new metric here
  - _self_
  ```


### 📚 3. Update the Evaluator

After implementing the new metric, you need to make sure, that `mymetric` can be used by `src/evaluation/evaluate_inference.py`. For this you need to add another elif-branch in `evaluate_inference(...){...}` so that the evaluator knows how to call the new metric.

*Example: Elif-Branch*
```python
elif metric_name == "mymetric":
    score = compute_mymetric(...)
```

 

## 📈 Expected Behavior
- The runner will recognize the new metric via Hydra.
- Results are appended to metrics.csv for each run.