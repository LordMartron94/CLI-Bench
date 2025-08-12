import dataclasses
from pathlib import Path
from typing import Callable, List

from .python_benchmark_command import PythonBenchmarkCommand
from .python_compare_command import PythonCompareCommand
from .python_profile_command import PythonProfileCommand
from ..common.py_common.cli_framework import CommandLineInterface
from ..common.py_common.logging import HoornLogger
from ..common.py_common.user_input.user_input_helper import UserInputHelper


@dataclasses.dataclass
class BenchmarkSuite:
    """Represents a named group of functions to be benchmarked together."""
    name: str
    functions: List[Callable]


@dataclasses.dataclass
class PythonCliContext:
    """Holds the global configuration for the Python benchmarking tool."""
    benchmark_root_dir: Path
    suites: List[BenchmarkSuite]
    exit_command: Callable

    bench_repeat: int = 30
    """The number of times to repeat each benchmark."""

    bench_number: int = 1000
    """The number of times the function is executed within each benchmark iteration."""


class PythonCLI:
    """
    The main CLI for the benchmarking tool. Manages the selection of benchmark
    suites and launches sub-menus for specific actions.
    """
    def __init__(self, logger: HoornLogger, context: PythonCliContext):
        self._logger = logger
        self._separator = "Benchmarking.Python"
        self._context = context
        self._main_cli = CommandLineInterface(logger, exit_command=context.exit_command, log_module_sep=self._separator)
        self._user_input_helper = UserInputHelper(logger)

        # The main CLI has only one job: to let the user select a suite.
        self._main_cli.add_command(
            keys=["suite", "s"],
            description="Select a benchmark suite to work with.",
            action=self._select_and_launch_suite
        )

    def start(self) -> None:
        """Starts the main suite selection loop."""
        print("Welcome to the Python Benchmarking Tool.")
        print("Type '/suite' or '/s' to select a benchmark suite.")
        self._main_cli.start_listen_loop()

    def exit(self) -> None:
        """Exits the main CLI loop."""
        self._main_cli.exit_conversation_loop()

    def _select_and_launch_suite(self) -> None:
        """Presents a menu of available suites and launches a sub-cli for the chosen one."""
        if not self._context.suites:
            self._logger.error("No benchmark suites have been defined.", separator=self._separator)
            return

        prompt = "Please select a suite to work with:\n"
        prompt += "\n".join(f"\t{i}) {suite.name}" for i, suite in enumerate(self._context.suites))
        prompt += "\nEnter your choice:"

        def validator(choice: int) -> tuple[bool, str]:
            if 0 <= choice < len(self._context.suites):
                return True, ""
            return False, f"Invalid choice. Must be between 0 and {len(self._context.suites) - 1}."

        try:
            choice = self._user_input_helper.get_user_input(
                prompt=prompt,
                expected_response_type=int,
                validator_func=validator
            )

            selected_suite = self._context.suites[choice]
            self._launch_suite_cli(selected_suite)

        except (ValueError, TypeError, KeyboardInterrupt):
            self._logger.info("\nSuite selection cancelled. Returning to main menu.", separator=self._separator)
            return

    def _launch_suite_cli(self, suite: BenchmarkSuite) -> None:
        """Creates and runs a temporary sub-menu tailored to the selected suite."""
        self._logger.info(f"--- Entering Suite: {suite.name} ---", separator=self._separator)

        suite_results_dir = self._context.benchmark_root_dir / suite.name

        suite_cli = CommandLineInterface(self._logger, log_module_sep=suite.name)
        suite_cli.set_exit_command(suite_cli.exit_conversation_loop)

        profile_command = PythonProfileCommand(
            self._logger,
            profile_options=suite.functions,
            benchmark_dir=suite_results_dir
        )
        benchmark_command = PythonBenchmarkCommand(
            self._logger,
            benchmark_functions=suite.functions,
            category_dir=suite_results_dir,
            repeat=self._context.bench_repeat,
            number=self._context.bench_number
        )
        compare_command = PythonCompareCommand(self._logger, suite_results_dir)

        suite_cli.add_command(["profile", "p"], "Profile all functions in this suite.", profile_command.run)
        suite_cli.add_command(["benchmark", "b"], "Run all benchmarks for this suite.", benchmark_command.run)
        suite_cli.add_command(["compare", "c"], "Compare results within this suite.", compare_command.run)

        suite_cli.start_listen_loop()

        self._logger.info(f"--- Exiting Suite: {suite.name} ---", separator=self._separator)
