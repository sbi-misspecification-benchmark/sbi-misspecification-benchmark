# SBI Benchmarking Framework

We want to use the top level script run.py as an entry point to launch the entire benchmarking process with a single command. As for now, by calling benchmark_run.py, it performs inference and evaluates it. Future additions include consolidations and visualisation of the results. 

---

## Running Benchmarks

###  Basic Usage

To run a single benchmark with default settings:

```bash
python -m src.run --config-path configs --config-name main
```
This will use configs/main.yaml to configure the task, inference method, number of simulations, and other parameters.

### Multirun
In order to launch multiple experiments, you can use Hydra’s --multirun mode
```bash
python -m src.run --multirun \
  inference=npe,nle \
  metric=c2st,ppc \
  inference.num_simulations=100,200
```
This command launches 8 runs (2 inference methods × 2 metrics × 2 simulation counts), each with a separate output directory under outputs/


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
```bash
outputs/
└── <TaskName>_<Method>/
    └── sims_<NumSimulations>/
        └── obs_<Index>/
            ├── posterior_samples.pt
            └── config_used.yaml
```


