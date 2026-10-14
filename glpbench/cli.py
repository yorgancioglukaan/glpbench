"""Top-level command-line interface for GLPBench."""

import argparse
import sys


_COMMANDS = {
    "experiment": ("main", "Run one link-prediction experiment."),
    "benchmark": ("benchmark", "Run a benchmark configuration."),
    "search": ("search", "Run validation-only hyperparameter search."),
    "audit": ("convergence_audit", "Run a convergence audit."),
}


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

    module_name, _ = _COMMANDS[args.command]
    original_argv = sys.argv
    sys.argv = [f"glpbench {args.command}", *args.arguments]
    try:
        if module_name == "main":
            import main as experiment_runner

            run_id = experiment_runner.run_experiment(
                experiment_runner.parse_args()
            )
            print(f"Completed experiment run {run_id}")
        else:
            module = __import__(module_name)
            module.main()
    finally:
        sys.argv = original_argv
    return 0

