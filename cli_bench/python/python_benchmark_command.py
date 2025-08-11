import json
import statistics
import timeit
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any

from ..benchmark_command_interface import IBenchmarkCommand
from ..common.py_common.logging import HoornLogger
from ..common.py_common.user_input.user_input_helper import UserInputHelper

class PythonBenchmarkCommand(IBenchmarkCommand):
    """
    Runs benchmarks and saves detailed statistical results to a JSON file.
    """
    def __init__(
            self,
            logger: HoornLogger,
            benchmark_options: Dict[str, Callable],
            benchmark_results_dir: Path,
            repeat: int = 30,
            number: int = 1000
    ) -> None:
        super().__init__(logger, is_child=True)
        self._separator = "Python"
        self._benchmark_options = benchmark_options
        self._results_dir = benchmark_results_dir
        self._repeat = repeat
        self._number = number
        self._user_input_helper = UserInputHelper(logger)
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def run(self) -> None:
        """Presents a menu, gets an output name, and runs the selected benchmark."""
        category, fn = self._get_user_choice()
        if not category:
            return

        output_name = self._user_input_helper.get_user_input(
            "Enter the output name for the benchmark results (e.g., 'list_comp_v1'):",
            expected_response_type=str,
            validator_func=lambda x: (bool(x), "Output name cannot be empty.")
        )
        output_path = self._results_dir.joinpath(f"{output_name}.json")
        self._results_dir.mkdir(exist_ok=True)

        self._execute_and_save_benchmark(fn, category, output_path)

    def _get_user_choice(self) -> tuple[str | None, Callable | None]:
        def __validate_choice(choice: int, max_amount: int) -> tuple[bool, str]:
            if 0 < choice <= max_amount:
                return True, ""
            return False, "Invalid choice."

        option_amount = len(self._benchmark_options)
        prompt = "Benchmarking Options:"
        lookup: Dict[int, tuple[str, Callable]] = {}
        for i, (name, fn) in enumerate(self._benchmark_options.items()):
            prompt += f"\n\t- {i + 1}) '{name}'"
            lookup[i + 1] = (name, fn)
        prompt += "\nChoose the benchmark to execute."

        choice = self._user_input_helper.get_user_input(
            prompt,
            expected_response_type=int,
            validator_func=lambda c: __validate_choice(c, option_amount)
        )

        return lookup[choice]

    def _execute_and_save_benchmark(self, fn: Callable, category: str, output_path: Path) -> None:
        """Runs the benchmark, prints stats, and saves the detailed results."""
        self._logger.info(f"Running benchmark for '{category}'...", separator=self._separator)
        self._logger.info(f"Configuration: {self._repeat} runs, {self._number} loops each.", separator=self._separator)

        timings = timeit.repeat(stmt=fn, repeat=self._repeat, number=self._number)
        timings_per_op = [t / self._number for t in timings]

        # --- Calculate Statistics ---
        mean = statistics.mean(timings_per_op)
        stdev = statistics.stdev(timings_per_op)
        median = statistics.median(timings_per_op)
        min_val, max_val = min(timings_per_op), max(timings_per_op)

        # --- Log Results to Console ---
        self._logger.info("--- Benchmark Results ---", separator=self._separator)
        self._logger.info(f"Mean (+/- std dev): {mean*1e6:.2f} µs/op (+/- {stdev*1e6:.2f})", separator=self._separator)
        self._logger.info(f"Median:             {median*1e6:.2f} µs/op", separator=self._separator)
        self._logger.info(f"Range (min-max):    [{min_val*1e6:.2f} - {max_val*1e6:.2f}] µs/op", separator=self._separator)
        self._logger.info("-------------------------", separator=self._separator)

        # --- Prepare data for JSON export ---
        results_data = {
            "metadata": {
                "name": category,
                "timestamp_utc": datetime.utcnow().isoformat(),
                "repeat": self._repeat,
                "number_per_repeat": self._number,
            },
            "stats": {
                "unit": "seconds_per_op",
                "mean": mean,
                "stdev": stdev,
                "median": median,
                "min": min_val,
                "max": max_val,
            },
            "timings": timings_per_op,
        }

        # --- Save to file ---
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        self._logger.info(f"Benchmark results saved to: {output_path}", separator=self._separator)
