import dataclasses
from pathlib import Path
from typing import Callable, Dict

from .python_benchmark_command import PythonBenchmarkCommand
from .python_compare_command import PythonCompareCommand
from .python_profile_command import PythonProfileCommand
from ..common.py_common.cli_framework import CommandLineInterface
from ..common.py_common.logging import HoornLogger

@dataclasses.dataclass
class PythonCommandContext:
    benchmark_dir: Path
    """The path where to save benchmark results."""

    options: Dict[str, Callable]
    """The functions one can profile, bench, etc., along with their category."""

    exit_command: Callable
    """The command to execute when exiting."""

    bench_iterations: int = 30
    """The number of iterations to run when benchmarking."""

    bench_number: int = 1000
    """The number of times to execute the code in each benchmark iteration."""


class PythonCLI:
    def __init__(self, logger: HoornLogger, command_context: PythonCommandContext):
        self._cli_interface = CommandLineInterface(logger, exit_command=command_context.exit_command)
        self._logger: HoornLogger = logger

        commands_to_add = {
            0: {
                "keys": ["profile", "p"],
                "description": "Starts the profiling suite.",
                "command": PythonProfileCommand(logger, command_context.options, command_context.benchmark_dir).run,
            },
            1: {
                "keys": ["benchmark", "b"],
                "description": "Starts the benchmarking suite.",
                "command": PythonBenchmarkCommand(logger, command_context.options, command_context.benchmark_dir, command_context.bench_iterations, command_context.bench_number).run
            },
            2: {
                "keys": ["benchmark-compare", "bc"],
                "description": "Compares benchmarks.",
                "command": PythonCompareCommand(logger, command_context.benchmark_dir).run
            },
        }

        for _, command_info in commands_to_add.items():
            self._cli_interface.add_command(command_info["keys"], command_info["description"], command_info["command"])

    def start(self) -> None:
        self._cli_interface.start_listen_loop()

    def exit(self):
        self._cli_interface.exit_conversation_loop()
