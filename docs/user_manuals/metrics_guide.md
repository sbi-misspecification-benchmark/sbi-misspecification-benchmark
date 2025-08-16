# Adding a New Metric

A metric evaluates how close inferred posteriors are to ground-truth reference distributions.

## Implementation and Example

### 1. Set up a Python file
If you want to add a new metric, create a new .yaml config file a new python file `src/evaluation/metrics/mymetric.py`.

Example:
<pre> def compute_mymetric(posterior_samples, reference_samples):
    return float(abs(posterior_samples.mean() - reference_samples.mean()))</pre> 

This simple metric measures the absolute difference in means between posterior and reference distributions.


**Note**: Since metrics can differ significantly in their logic (e.g. C2ST trains a classifier, while PPC generates data and compares it to observations), there is no shared interface that all metrics must implement.       
However, each metric should implement the main function named `compute_mymetric(...)` that takes the necessary inputs (e.g. posterior samples, reference data) and returns a scalar score.

### 2. Update the Evaluator

After implementing the new metric, you need to make sure, that `mymetric` can be used by `src/evaluation/evaluate_inference.py`. For this you need to add another elif-branch  in `evaluate_inference(...)` so that the evaluator knows how to call the new metric.

Elif-Branch:
<pre>if metric_name == "mymetric":
    score = compute_mymetric(...)</pre>

### 3. Add a new Config File
For it to be recognised by Hydra you then need to add a new .yaml config file `src/configs/metric/mymetric.yaml`, that follows the simple structure:

Example:    
<pre>name: mymetric</pre> 

## Expected Behavior
- The metric becomes selectable with metric=mymetric in the main config file `main.yaml`in Hydra.
- Results are appended to metrics.csv for each run.