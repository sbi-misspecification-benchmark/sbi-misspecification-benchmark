import argparse

def run_tool():
    """Placeholder function for the 'run' command."""
    return "Running the tool..."

def list_methods():
    """Placeholder function for the 'list-methods' command."""
    methods = ["Method 1: ExampleMethodA", "Method 2: ExampleMethodB"]
    return "\n".join(methods)

def main():
    parser = argparse.ArgumentParser(
        description="CLI for interacting with the sbi-misspecification-benchmark tool."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("run", help="Run the tool.")

    subparsers.add_parser("list-methods", help="List available methods.")

    args = parser.parse_args()

    if args.command == "run":
        print(run_tool())
    elif args.command == "list-methods":
        print(list_methods())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()