import argparse
valid_metrics = ["ppc", "mmd", "c2st", "probability_true"]

def run_tool(): # dummy
    """Placeholder function for the 'run' command."""
    return "Running the tool..."

def list_methods(): # dummy
    """Placeholder function for the 'list-methods' command."""
    methods = ["Method 1: ExampleMethodA", "Method 2: ExampleMethodB"]
    return "\n".join(methods)

def ask_user_for_metrics():
    valid_metrics = ["ppc", "mmd", "c2st", "probability_true"]
    selected = list()

    print("You can now choose the metrics to be included")
    for metric in valid_metrics:
        answer = input(f"{metric}? (y/n): ").strip().lower()
        if answer == "y":
            selected.append(metric)
        elif answer == "n":
            continue
        else:
            print(f"{answer} is not a valid input.")
    return selected

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="CLI for interacting with the sbi-misspecification-benchmark tool."
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the tool.")
    run_parser.add_argument(
        "--metrics",
        type = str,
    )
    # list-methods command
    subparsers.add_parser("list-methods", help="List available methods.")


    # Parse arguments
    args = parser.parse_args()

    # Command handling
    if args.command == "run":
        print("Enter metrics or press Enter to choose")
        metrics_input = input(args.metrics or "").strip().lower()
        if metrics_input:
            user_metrics = [m.strip() for m in metrics_input.split(",")]

            invalid = [m for m in user_metrics if m not in valid_metrics]

            if invalid:
                print(f"Invalid metrics: {', '.join(invalid)}")
                print(f"Valid metrics are: {', '.join(valid_metrics)}")
            else:
                print(f"Running the tool with metrics: {', '.join(user_metrics)}")
        else:
            selected = ask_user_for_metrics()
            if not selected:
                print("No metrics selected.")
            else:
                print(f"Running the tool with metrics: {', '.join(sorted(selected))}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()