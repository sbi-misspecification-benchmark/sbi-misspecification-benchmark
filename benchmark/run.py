import argparse
import yaml
import random

def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks using a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Handle random seed
    random_seed = config.get('random_seed')
    if random_seed is None:
        random_seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {random_seed}")
    else:
        print(f"Using provided random seed: {random_seed}")

    # Process task
    task = config.get('task')  
    if task is not None:
        task_name = task.get('name')
    else:
        task_name = None

print(f"Executing task: {task_name}")
    # Add logic to handle task execution => num_simulations, num_observations, num_posterior_samples


if __name__ == "__main__":
    main()