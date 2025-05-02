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
    task = config.get('task')
    if task is not None:
        task_name = task.get('name')
        # Get keys for testing purposes w/o defaults:
        num_simulations = task.get('num_simulations')
        num_observations = task.get('num_observations')
        num_posterior_samples = task.get('num_posterior_samples')
    else:
        task_name = "No task provided"

    print(
        f"Executing task: {task_name}\n"
        # printing keys for testing purposes
        f"num_simulations: {num_simulations}\n"
        f"num_observations: {num_observations}\n"
        f"num_posterior_samples: {num_posterior_samples}\n"
    )

    # Add logic to handle task execution => num_simulations, num_observations, num_posterior_samples

if __name__ == "__main__":
    main()