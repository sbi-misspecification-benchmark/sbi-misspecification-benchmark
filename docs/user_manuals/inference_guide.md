# Adding a New Inference Method

An inference method specifies how posterior distributions are learned or approximated given simulated data.



## Implementation of a New Inference Method
- In this project we only used the methods that are already implemented in the SBI tool.
- If you want to add another method from the SBI tool, register it in `src/inference/Run_Inference.py` inside the methods dictionary. 

Methods Dictionary in `src/inference/Run_Inference.py` should have this structure:
<pre> methods = {
    "NPE": NPE,
    "MYMETHOD": MyMethod,
}</pre> 

The key is the config name. The value is the SBI method that performs inference.

Furthermore you have to add a new .yaml config file in `src/configs/inference/`.



## How to set up a New Config
Add a new .yaml config file `src/configs/inference/mymethod.yaml` that specifies the parameters method, num_simulations, num_observations and num_posterior_samples. 

Example:    
<pre>method: mymethod 
num_simulations: 500,1000 
num_observations: 4 
num_posterior_samples: 100 </pre>  

This config defines the method name and key hyperparameters. For a Hydra Multirun add multiple values (e.g. num_simlations: 500, 1000), for a Singlerun, you only need one value. 



## Expected Behavior
- run_inference will train the chosen method and save posterior_samples.pt files.
- The config used for the run will be stored in the output folder for reproducibility.