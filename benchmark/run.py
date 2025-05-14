import argparse
import yaml
import random

# Define a dummy task class
class DummyTask:
    def simulator(self, thetas):
        print(f"Running DummyTask simulator with thetas: {thetas}")

# Task registry to hold all available task classes
task_registry = {
    "test_task": DummyTask
}

def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def validate_positive(value, default_value):
    """Ensure a configuration value is non-negative, otherwise use the default."""
    if value is None or value < 0:
        return default_value
    return value

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks using a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Handle random seed
    random_seed = config.get('random_seed')
    if random_seed is None:
        random_seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {random_seed}")
    else:
        print(f"Using provided random seed: {random_seed}")

    # Process task
    task = config.get('task', {})
    task_name = task.get('name', "Default Task Name")

    # Check if the task exists in the registry
    if task_name not in task_registry:
        raise ValueError(
            f"Task '{task_name}' not found in the registry. "
            f"Available tasks are: {list(task_registry.keys())}"
        )

    num_simulations = validate_positive(task.get('num_simulations'), 100)  # Defaults to 100 simulations
    num_observations = validate_positive(task.get('num_observations'), 10)  # Defaults to 10 observations
    num_posterior_samples = validate_positive(task.get('num_posterior_samples'), 50)  # Defaults to 50 posterior samples

    method = config.get('method', "default_method")  # Default method if not provided

    print(
        f"Executing task: {task_name}\n"
        f"num_simulations: {num_simulations}\n"
        f"num_observations: {num_observations}\n"
        f"num_posterior_samples: {num_posterior_samples}\n"
        f"method: {method}"
    )

    # Add logic to handle task execution => num_simulations, num_observations, num_posterior_samples
    # Placeholder for task execution
    # For now, we simulate task execution using the dummy simulator
    
    # Instantiate the task
    task_class = task_registry[task_name]
    task_instance = task_class()

    # Placeholder for thetas
    thetas = "misspecified_model_parameters"

    # Simulate the task
    task_instance.simulator(thetas)


if __name__ == "__main__":
    main()