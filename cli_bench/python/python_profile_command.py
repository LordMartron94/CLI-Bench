from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, TypeVar, Tuple

from viztracer import VizTracer

from ..benchmark_command_interface import IBenchmarkCommand
from ..common.py_common.logging import HoornLogger
from ..common.py_common.user_input.user_input_helper import UserInputHelper

T = TypeVar('T')

def run_with_profiling(func: Callable, category: str, benchmark_dir: Path) -> None:
    """
    Runs the given function under VizTracer profiling and writes a benchmark file.

    :param func: A no-argument function to execute and profile.
    :param category: The profiling category (used to name the subdirectory).
    :return: The return value of the function.
    """
    # Prepare timestamp and paths
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d-%H_%M_%S")
    benchmark_dir = benchmark_dir.joinpath(category)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = benchmark_dir.joinpath(f"{timestamp}-benchmark.json")

    # Run with VizTracer
    with VizTracer(
            output_file=str(benchmark_path),
            min_duration=1,
            ignore_c_function=True,
            ignore_frozen=True
    ) as _:
        func()

    return


class PythonProfileCommand(IBenchmarkCommand):
    def __init__(self, logger: HoornLogger, profile_options: Dict[str, Callable], benchmark_dir: Path) -> None:
        super().__init__(logger, is_child=True)
        self._separator: str = f"Python"

        self._profile_options: Dict[str, Callable] = profile_options
        self._benchmark_dir = benchmark_dir
        self._user_input_helper: UserInputHelper = UserInputHelper(logger)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def run(self) -> None:
        def __validate_choice(choice: int, max_amount: int) -> Tuple[bool, str]:
            if 0 < choice <= max_amount:
                return True, ""
            return False, "Invalid choice."

        option_amount: int = len(self._profile_options)

        prompt: str = "Profiling Options:"

        lookup: Dict[int, Tuple[str, Callable]] = {}
        for i, (name, fn) in enumerate(self._profile_options.items()):
            prompt += f"\n\t- {i + 1}) '{name}'"
            lookup[i+1] = (name, fn)

        prompt += "\nChoose the profile to execute."

        choice = self._user_input_helper.get_user_input(prompt, expected_response_type=int, validator_func=lambda c: __validate_choice(c, option_amount))
        category, fn = lookup[choice]

        self._execute_with_viztracer(fn, category)

    def _execute_with_viztracer(self, fn: Callable, category: str, ) -> None:
        run_with_profiling(fn, category, benchmark_dir=self._benchmark_dir)
