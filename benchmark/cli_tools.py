import argparse

def main():
    parser = argparse.ArgumentParser(
        description="CLI for interacting with the sbi-misspecification-benchmark tool."
    )

    args = parser.parse_args()

if __name__ == "__main__":
    main()