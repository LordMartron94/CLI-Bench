import json
import re
import statistics
import timeit
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, List

from ..benchmark_command_interface import IBenchmarkCommand
from ..common.py_common.logging import HoornLogger
from ..common.py_common.user_input.user_input_helper import UserInputHelper


class PythonBenchmarkCommand(IBenchmarkCommand):
    """
    Runs benchmarks for a list of functions under a specified category
    and saves detailed statistical results to automatically named JSON files.
    """
    def __init__(
            self,
            logger: HoornLogger,
            benchmark_functions: List[Callable],
            category_dir: Path,
            repeat: int = 30,
            number: int = 1000
    ) -> None:
        super().__init__(logger, is_child=True)
        self._separator = "Benchmarking.Python"
        self._benchmark_functions = benchmark_functions
        self._category_dir = category_dir
        self._repeat = repeat
        self._number = number
        self._user_input_helper = UserInputHelper(logger)
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def run(self) -> None:
        """Runs all benchmarks for the functions provided at initialization."""
        self._category_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(f"Running benchmarks for category: '{self._category_dir.name}'", separator=self._separator)

        for func in self._benchmark_functions:
            raw_name = func.__name__
            sanitized_name = re.sub(r'[<>:"/\\|?*]', '_', raw_name)
            if not sanitized_name or sanitized_name == '_lambda_':
                self._logger.error(f"Cannot create a benchmark for an unnamed or lambda function. Please wrap it in a named function.", separator=self._separator)
                continue

            output_path = self._category_dir / f"{sanitized_name}.json"

            self._execute_and_save_benchmark(func, sanitized_name, output_path)

    def _execute_and_save_benchmark(self, fn: Callable, name: str, output_path: Path) -> None:
        self._logger.info(f"Running benchmark for '{name}'...", separator=self._separator)

        timings = timeit.repeat(stmt=fn, repeat=self._repeat, number=self._number)
        timings_per_op = [t / self._number for t in timings]

        mean = statistics.mean(timings_per_op)
        stdev = statistics.stdev(timings_per_op)
        median = statistics.median(timings_per_op)
        min_val, max_val = min(timings_per_op), max(timings_per_op)

        self._logger.info(f"Mean (+/- std dev): {mean*1e6:.2f} µs/op (+/- {stdev*1e6:.2f})", separator=self._separator)

        results_data = {
            "metadata": {
                "name": name,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "repeat": self._repeat,
                "number_per_repeat": self._number,
            },
            "stats": { "unit": "seconds_per_op", "mean": mean, "stdev": stdev, "median": median, "min": min_val, "max": max_val, },
            "timings": timings_per_op,
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        self._logger.info(f"Benchmark results for '{name}' saved to: {output_path}", separator=self._separator)
