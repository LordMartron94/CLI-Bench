import json
from pathlib import Path
from typing import List, Dict, Any

from scipy.stats import ttest_ind

from ..benchmark_command_interface import IBenchmarkCommand
from ..common.py_common.logging import HoornLogger
from ..common.py_common.user_input.user_input_helper import UserInputHelper


class PythonCompareCommand(IBenchmarkCommand):
    """
    Compares two or more saved benchmark JSON files and reports statistical significance against a baseline.
    """
    def __init__(self, logger: HoornLogger, benchmark_results_dir: Path):
        super().__init__(logger, is_child=True)
        self._separator = "Python"
        self._results_dir = benchmark_results_dir
        self._user_input_helper = UserInputHelper(logger)
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def run(self) -> None:
        """The main execution logic for comparing benchmarks."""
        result_files = sorted([p for p in self._results_dir.glob("*.json")])
        if not result_files:
            self._logger.error(f"No benchmark results found in '{self._results_dir}'.", separator=self._separator)
            return

        # 1. Get multiple user selections in one go
        selected_files = self._get_user_selections(result_files)
        if len(selected_files) < 2:
            self._logger.warning("Please select at least two benchmarks to compare.", separator=self._separator)
            return

        # 2. Load all selected benchmark data
        all_data = []
        for file in selected_files:
            with open(file) as f:
                # Store the data and its original filename
                all_data.append({"data": json.load(f), "name": file.name})

        # 3. The first selection is the baseline, the rest are for comparison
        baseline = all_data[0]
        comparisons = all_data[1:]

        # 4. Perform comparison and print the consolidated table
        self._compare_and_print(baseline, comparisons)

    def _get_user_selections(self, files: List[Path]) -> List[Path]:
        """Presents a list of files and gets multiple user choices from a single input."""
        prompt = "Available benchmark results:\n"
        prompt += "\n".join(f"\t{i}) {file.name}" for i, file in enumerate(files))
        prompt += "\nEnter the numbers of the benchmarks to compare, separated by spaces."
        prompt += "\nThe first number will be the baseline."

        def validator(input_str: str) -> tuple[bool, str]:
            try:
                indices_str = input_str.strip().split()
                if not indices_str:
                    return False, "Input cannot be empty."

                indices = [int(i) for i in indices_str]

                if len(indices) != len(set(indices)):
                    return False, "Duplicate selections are not allowed."

                for i in indices:
                    if not (0 <= i < len(files)):
                        return False, f"Invalid choice: {i}. Must be between 0 and {len(files) - 1}."

                return True, ""
            except ValueError:
                return False, "Invalid input. Please enter numbers separated by spaces."

        chosen_str = self._user_input_helper.get_user_input(prompt, expected_response_type=str, validator_func=validator)

        chosen_indices = [int(i) for i in chosen_str.strip().split()]
        return [files[i] for i in chosen_indices]

    def _compare_and_print(self, baseline: Dict[str, Any], comparisons: List[Dict[str, Any]]) -> None:
        """Performs statistical comparison against a baseline and prints a formatted table."""
        baseline_data = baseline["data"]
        baseline_name = baseline["name"]
        baseline_mean = baseline_data["stats"]["mean"]
        baseline_timings = baseline_data["timings"]

        # --- Print Header ---
        header = f"{'Benchmark':<40} {'Time/op':>15} {'Delta':>15} {'P-value':>12}"
        self._logger.info("-" * len(header), separator=self._separator)
        self._logger.info(header, separator=self._separator)
        self._logger.info("-" * len(header), separator=self._separator)

        # --- Print Baseline Row ---
        baseline_time_str = f"{baseline_mean * 1e6:.2f}µs"
        baseline_row = f"{baseline_name:<40} {baseline_time_str:>15} {'(baseline)':>15} {'':>12}"
        self._logger.info(baseline_row, separator=self._separator)

        # --- Print Comparison Rows ---
        for item in comparisons:
            comp_data = item["data"]
            comp_name = item["name"]
            comp_mean = comp_data["stats"]["mean"]
            comp_timings = comp_data["timings"]

            _, p_value = ttest_ind(baseline_timings, comp_timings, equal_var=False)

            percent_change = ((comp_mean - baseline_mean) / baseline_mean) * 100

            # --- Format result strings ---
            time_str = f"{comp_mean * 1e6:.2f}µs"
            delta_str = f"{percent_change:+.2f}%"
            pval_str = f"{p_value:.3f}"

            significance = "~" if p_value > 0.05 else " "

            row = f"{comp_name:<40} {time_str:>15} {delta_str:>14}{significance} {pval_str:>12}"
            self._logger.info(row, separator=self._separator)

        self._logger.info("-" * len(header), separator=self._separator)
