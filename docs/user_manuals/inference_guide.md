# Adding a New Inference Method

An inference method specifies how posterior distributions are learned or approximated given simulated data.


## Implementation and Example
### 🐍 1. Set up a Python file
In this project we only used the methods that are already implemented in the SBI tool. If you wanted to implement a new method, you can add a python file in `src\inference\methods`.


### 📜 2. Add a new Config File
Add a new .yaml config file `src/configs/inference/mymethod.yaml` that specifies the parameters method, num_simulations, num_observations and num_posterior_samples. 

*Example: mymethod.yaml*   
```yaml
method: mymethod 
num_simulations: 500,1000 
num_observations: 4 
num_posterior_samples: 100
``` 

This config defines the method name and key parameters.   
For a Hydra Multirun add multiple values (e.g. num_simlations: 500, 1000), for a Singlerun, you only need one value. 

Now you can call the new metric in the main config file `main.yaml`.   

*Example: Main.yaml Configuartion*
```yaml
defaults:
  - task: misspecified_likelihood     
  - inference: mymethod      # Call new method here
  - metric: c2st       
  - _self_
  ```






### 📚 3. Update the Registry
If you want to add another method from the SBI tool, register it in `src/inference/Run_Inference.py` inside the methods dictionary, so that the runner can find and call the new method. 

*Example: Method Registry in Run_Inference.py with MyMethod*
```python 
methods = {
    "NPE": NPE,
    "MYMETHOD": MyMethod,
}
```

The key is the config name. The value is the SBI method that performs inference.



## 📈 Expected Behavior
- run_inference will train the chosen method and save posterior_samples.pt files.
- The config used for the run will be stored in the output folder for reproducibility.