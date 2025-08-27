# Function: `compute_c2st(inference_samples, reference_samples, test_size, random_state, plot=False, obs_idx=None)`

This function implements the Classifier Two-Sample Test (C2ST), which measures how well a classifier distinguishes between posterior samples obtained from inference and ground-truth posterior samples. This allows to make a statement about the degree of misspecification.  

 

---

## Interpretation of Scores

- **~0.5** → The classifier cann't distinguish between inference and reference samples → no or low misspecification   
- **>0.5** → Classifier can distinguish → larger misspecification between inference and reference  
- **Close to 1.0** → Samples are clearly different → inference didn't generate the true posterior  

---

## Example Usage

```python
accuracy = compute_c2st(
    inference_samples=inference_samples.numpy(),
    reference_samples=reference_samples.numpy(),
    test_size=0.3,
    random_state=42,
    plot=True,
    obs_idx=1
)
```
---
## Additional functionality

For 2-dimensional data it is possible to plot the samples. Set plot=True, obs_idx ist used for the titles of the plots.
