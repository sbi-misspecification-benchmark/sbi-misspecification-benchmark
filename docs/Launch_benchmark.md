# SBI Benchmarking Framework

We want to use the top level script run.py as an entry point to launch the entire benchmarking process with a single command. As for now, by calling benchmark_run.py, it performs inference and evaluates it. Future additions include consolidations and visualisation of the results. 

---

## Running Benchmarks

###  Basic Usage

To run the benchmark with default settings:

```bash
python -m src.run --config-path configs --config-name main
```
This will use configs/main.yaml to configure the task, inference method, number of simulations, and other parameters.




## Structure
The benchmark is composed of the following components:

### Entry Point: run.py
This is the main entry point that:

loads the configuration using Hydra
calls the benchmarking logic in benchmark_run.py

### Core Logic: benchmark_run.py
This module handles:

task initialization (e.g., likelihood-misspecified models)
calling run_inference(...) to generate posteriors and saving them 
evaluating results with evaluate_inference(...)

### Output structure
the benchmark produces two types of outputs:
## 1.
posterior samples and the config used for each observation
```bash
outputs/
└── <TaskClassName>_<Method>/
    └── sims_<NumSimulations>/
        └── obs_<Index>/
            ├── posterior_samples.pt
            └── config_used.yaml

outputs/LikelihoodMisspecifiedTask_NPE/
└── sims_100/
    ├── obs_0/
    │   ├── posterior_samples.pt
    │   └── config_used.yaml
    ├── obs_1/
    │   ├── posterior_samples.pt
    │   └── config_used.yaml
    ...



```
## 2.
a metrics.csv file that accumulates evaluation scores across all observations for a given number of simulations.
```bash
outputs/
└── <ConfigTaskName>_<Method>_Solution/
    └── sims_<NumSimulations>/
        └── metrics.csv
outputs/misspecified_likelihood_NPE_Solution/
└── sims_100/
    └── metrics.csv



```


