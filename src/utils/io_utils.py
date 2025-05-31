import os
import torch
#utils for i/o that are needed for evaluate_inference

# saves posterior samples as torch tensor
# in a standardized path
def save_samples(tensor, task_name, method_name):
    path = f"outputs/{task_name}/{method_name}/posterior_samples.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)

# saves reference posterior samples as torch tensor
# in a standardized path
def save_reference_samples(tensor, task_name, seed):
    path = f"outputs/{task_name}/reference/ref_posterior_seed{seed}.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)

# loads a tensor to be used for evaluation
def load_tensor(task_name, method_name, name):
    return torch.load(f"outputs/{task_name}/{method_name}/{name}.pt")

# saves a panda dataframe as CSV file
def save_file(task_name, method_name, file_name, dataframe):
    path = f"outputs/{task_name}/{method_name}/{file_name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataframe.to_csv(path, index=False)

