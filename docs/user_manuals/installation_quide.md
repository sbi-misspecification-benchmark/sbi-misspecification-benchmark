# Installation Guide

## Prerequisites
- Python 3.10: Required for compatibility with the SBI package dependencies
- [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html): Recommended package manager to isolate project dependencies.

## Standard Setup 
### 1. Create Conda Environment
    conda create -n sbi-miss-bench python=3.10 -y
   
Environment isolation prevents version conflicts between different projects on your system.

### 2. Activate the environment
    conda activate sbi-miss-bench
 
Switches your terminal session to use the newly created environment.  
You'll need to activate this environment every time you open a new terminal window to work on the project.

### 3. Install the sbi package
    python -m pip install sbi 

This installs the main SBI package from PyPI along with all necessary dependencies like PyTorch, NumPy, and SciPy.  

### 4. Clone the repository
    git clone https://github.com/mackelab/sbi-misspecification-benchmark.git
    cd sbi-misspecification-benchmark
   
A local copy of the benchmark code and access to all project files.


