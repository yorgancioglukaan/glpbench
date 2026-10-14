"""Top-level command-line interface for GLPBench."""

import argparse
import sys

_COMMANDS = {
    "experiment": "Run one link-prediction experiment.",
    "benchmark": "Run a benchmark configuration.",
    "search": "Run validation-only hyperparameter search.",
    "audit": "Run a convergence audit.",
}


def _command_module(command):
    """Import and return one package-native runner module."""
    if command == "experiment":
        from glpbench.runners import experiment

        return experiment
    if command == "benchmark":
        from glpbench.runners import benchmark

        return benchmark
    if command == "search":
        from glpbench.runners import search

        return search
    from glpbench.runners import audit

    return audit


def _parser():
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="glpbench",
        description=(
            "Evaluate graph link-prediction models under transductive and "
            "semi-inductive protocols."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=tuple(_COMMANDS),
        help="command to run",
    )
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="arguments passed to the selected command",
    )
    return parser


def main():
    """Dispatch a GLPBench command to the existing runner."""
    parser = _parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0

    module = _command_module(args.command)
    original_argv = sys.argv
    sys.argv = [f"glpbench {args.command}", *args.arguments]
    try:
        if args.command == "experiment":
            run_id = module.run_experiment(
                module.parse_args()
            )
            print(f"Completed experiment run {run_id}")
        else:
            module.main()
    finally:
        sys.argv = original_argv
    return 0

