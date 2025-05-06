"""Main module for running the antmaxxing package as a script."""

import sys

from .operations import add


def main() -> None:
    """Run the antmaxxing package from the command line.

    Handles basic CLI input for add operation.
    """
    expected_number_of_arguments = 4
    if len(sys.argv) != expected_number_of_arguments:
        print("Usage: python -m calculator [add|subtract] num1 num2")  # noqa: T201
        sys.exit(1)

    operation = sys.argv[1]
    a = float(sys.argv[2])
    b = float(sys.argv[3])

    if operation == "add":
        result = add(a, b)
    else:
        print(f"Unknown operation: {operation}")  # noqa: T201
        sys.exit(1)

    print(f"Result: {result}")  # noqa: T201


if __name__ == "__main__":
    main()
