import argparse

# evaluation used in Lueckmann et al. (2021) - Benchmarking Simulation-Based Inference
valid_metrics = ["ppc", "mmd", "c2st", "probability_true"]

def run_tool(args):  # dummy
    return f"Running the tool with evaluation: {', '.join(args)}"

def list_methods():  # dummy
    """Placeholder function for the 'list-methods' command."""
    methods = ["Method 1: ExampleMethodA", "Method 2: ExampleMethodB"]
    return "\n".join(methods)

def help_function(parser):
    # guides the user through the program
    print("CLI for interacting with the sbi-misspecification-benchmark tool.")
    print("Choose one of the following commands: \n  1) run\n  2) list-methods\n  3) help/exit")
    choice = input("Enter a number ").strip()
    if choice == "1":
        handle_command(argparse.Namespace(metrics=None))
    elif choice == "2":
        print(list_methods())
    elif choice == "3":
        parser.print_help()
        return
    else:
        print("Invalid choice.")
        help_function(parser)

def handle_command(args):
    # user can either enter valid evaluation manually or choose them
    if args.metrics:
        metrics_input = args.metrics.strip().lower()
    else:
        print("Enter evaluation or press Enter to choose interactively")
        metrics_input = input().strip().lower()

    if metrics_input:
        user_metrics = [m.strip() for m in metrics_input.split(",")]
        invalid = [m for m in user_metrics if m not in valid_metrics]

        if invalid:
            print(f"Invalid evaluation: {', '.join(invalid)}")
            print(f"Valid evaluation are: {', '.join(valid_metrics)}")
        else:
            print(run_tool(user_metrics))
    else:
        selected = ask_user_for_metrics()
        if not selected:
            print("No evaluation selected.")
        else:
            print(run_tool(selected))

def ask_user_for_metrics():
    # function to choose the evaluation
    selected = []
    print("You can now choose the evaluation to be included")
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
        "--evaluation",
        type=str,
    )

    # list-methods command
    subparsers.add_parser("list-methods", help="List available methods.")

    # Parse arguments
    args = parser.parse_args()

    # Command handling
    if args.command == "run":
        handle_command(args)
    elif args.command is None:
        help_function(parser)
    elif args.command == "list-methods":
        print(list_methods())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
