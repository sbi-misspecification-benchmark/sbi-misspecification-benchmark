import argparse

def run_tool(): # dummy
    """Placeholder function for the 'run' command."""
    return "Running the tool..."

def list_methods(): # dummy
    """Placeholder function for the 'list-methods' command."""
    methods = ["Method 1: ExampleMethodA", "Method 2: ExampleMethodB"]
    return "\n".join(methods)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="CLI for interacting with the sbi-misspecification-benchmark tool."
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    subparsers.add_parser("run", help="Run the tool.")

    # list-methods command
    subparsers.add_parser("list-methods", help="List available methods.")

    # Parse arguments
    args = parser.parse_args()

    # Command handling
    if args.command == "run":
        print(run_tool())
    elif args.command == "list-methods":
        print(list_methods())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()