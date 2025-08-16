# General Guide
This file explains the overall structure of the repository and how to run experiments.   
It is intended as an entry point for users who want to understand how everything connects.

## Project Structure and Important Files



- **Main entrypoint using Hydra.**     
*src/run.py*       
This script parses config files and launches benchmark runs.   

- **Benchmark loop and task registry.**   
*src/utils/benchmark_run.py*       
Tasks are instantiated and the overall simulation and evaluation flow is orchestrated.

- **Coordination of inference methods.**   
*src/inference/Run_Inference.py*    
Contains the `run_inference` function and the mapping from method names to implementations.

- **Metric coordination.**   
*src/evaluation/evaluate_inference.py*        
Calls the evaluation function depending on the chosen metric.

- **Metric definitions.**   
*src/evaluation/metrics/*        
Each metric lives in its own file (e.g. `c2st.py`). For more information see *metrics_guide.md*.

- **Task implementations.**   
*src/tasks/*      
Each task defines priors, simulators, and reference posteriors. For more information see *tasks_guide.md*.

- **Hydra config files.**   
*src/configs/*      
Subfolders contain YAML configs for tasks, inference methods, and metrics.



## Expected Outputs

After Hydra multirun jobs complete, the runner script `run.py` automatically consolidates results and creates visualizations:

- Consolidated metrics are stored in `outputs/{task}_{method}/metrics_all.csv`.
- Individual run metrics (`metrics.csv`) remain in their simulation folders.
- Plots generated from the consolidated metrics are saved under `outputs/{task}_{method}/plots/`.

This means that after a full benchmark run you will have:
- Posterior samples (`posterior_samples.pt`) for each observation index and simulation count.
- A merged CSV file (`metrics_all.csv`) summarizing all metrics of the whole run.
- A set of plots in the `plots/` directory.



## Run the Project
The entire benchmark process — including inference, evaluation, consolidation, and visualization — is launched via a single command like:
`python run.py --config-path=configs --config-name=main`