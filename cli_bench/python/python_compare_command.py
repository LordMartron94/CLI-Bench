import csv
import json
from pathlib import Path
from typing import List, Dict, Any

from scipy import stats
from scipy.stats import ttest_ind
from tabulate import tabulate

from ..benchmark_command_interface import IBenchmarkCommand
from ..common.py_common.logging import HoornLogger
from ..common.py_common.user_input.user_input_helper import UserInputHelper


class PythonCompareCommand(IBenchmarkCommand):
    """
    Compares two or more saved benchmark JSON files from a chosen category
    and reports statistical significance against a baseline.
    """
    def __init__(self, logger: HoornLogger, category_to_compare_dir: Path):
        super().__init__(logger, is_child=True)
        self._separator = "Benchmarking.Python"
        self._category_dir = category_to_compare_dir
        self._user_input_helper = UserInputHelper(logger)
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def run(self) -> None:
        """The main execution logic for comparing benchmarks within a given category."""
        self._logger.info(f"Analyzing benchmarks in category: '{self._category_dir.name}'", separator=self._separator)

        result_files = sorted([p for p in self._category_dir.glob("*.json")])
        if len(result_files) < 2:
            self._logger.error(f"Need at least 2 results in '{self._category_dir.name}' to compare.", separator=self._separator)
            return

        selected_files = self._get_user_selections(result_files)
        if len(selected_files) < 2:
            self._logger.warning("Please select at least two benchmarks to compare.", separator=self._separator)
            return

        all_data = []
        for file in selected_files:
            with open(file) as f:
                all_data.append({"data": json.load(f), "name": file.name})

        baseline = all_data[0]
        comparisons = all_data[1:]

        self._compare_and_print(baseline, comparisons)

    def _select_category(self, categories: List[Path]) -> Path:
        """Helper to present a list of directories and get a valid user choice."""
        prompt = "\n".join(f"\t{i}) {cat.name}" for i, cat in enumerate(categories))
        prompt += "\nChoose a category:"

        def validator(choice: int) -> tuple[bool, str]:
            if 0 <= choice < len(categories):
                return True, ""
            return False, "Invalid choice."

        choice_idx = self._user_input_helper.get_user_input(
            prompt, expected_response_type=int, validator_func=validator
        )
        return categories[choice_idx]

    def _get_user_selections(self, files: List[Path]) -> List[Path]:
        prompt = "Available benchmark results:\n"
        prompt += "\n".join(f"\t{i}) {file.name}" for i, file in enumerate(files))
        prompt += "\nEnter the numbers of the benchmarks to compare, separated by spaces."
        prompt += "\nThe first number will be the baseline."

        def validator(input_str: str) -> tuple[bool, str]:
            try:
                indices_str = input_str.strip().split()
                if not indices_str: return False, "Input cannot be empty."
                indices = [int(i) for i in indices_str]
                if len(indices) != len(set(indices)): return False, "Duplicate selections are not allowed."
                for i in indices:
                    if not (0 <= i < len(files)): return False, f"Invalid choice: {i}. Must be between 0 and {len(files) - 1}."
                return True, ""
            except ValueError:
                return False, "Invalid input. Please enter numbers separated by spaces."

        chosen_str = self._user_input_helper.get_user_input(prompt, expected_response_type=str, validator_func=validator)
        chosen_indices = [int(i) for i in chosen_str.strip().split()]
        return [files[i] for i in chosen_indices]

    # noinspection t
    def _compare_and_print(self, baseline: Dict[str, Any], comparisons: List[Dict[str, Any]]) -> None:
        """
        Performs statistical comparison against a baseline and prints a formatted table.
        Optionally exports results to a CSV file.
        """
        baseline_data = baseline["data"]
        baseline_name = baseline["name"]
        baseline_mean = baseline_data["stats"]["mean"]
        baseline_std = baseline_data["stats"].get("stdev", 0)
        baseline_timings = baseline_data["timings"]
        baseline_n = len(baseline_timings)

        # --- Dynamic Time Unit Selection ---
        def format_time(seconds: float) -> tuple[str, float]:
            if seconds >= 1:
                return f"{seconds:.2f}s", 1
            elif seconds >= 1e-3:
                return f"{seconds * 1e3:.2f}ms", 1e3
            else:
                return f"{seconds * 1e6:.2f}µs", 1e6

        # --- Calculate Baseline Confidence Interval ---
        confidence_level = 0.95
        if baseline_n > 1 and baseline_std > 0:
            sem = baseline_std / (baseline_n ** 0.5)
            ci_margin = stats.t.ppf((1 + confidence_level) / 2, baseline_n - 1) * sem
            baseline_ci_lower = baseline_mean - ci_margin
            baseline_ci_upper = baseline_mean + ci_margin
        else:
            baseline_ci_lower, baseline_ci_upper = baseline_mean, baseline_mean

        # --- Prepare Table Data ---
        table_data = []
        table_headers = ["Benchmark", "Time", "Std Dev", "Delta", "Factor", "P-value", "Signif", "95% CI"]

        # --- Baseline Row ---
        baseline_time_str, _ = format_time(baseline_mean)
        baseline_std_str = format_time(baseline_std)[0]
        baseline_ci_str = f"[{format_time(baseline_ci_lower)[0]}, {format_time(baseline_ci_upper)[0]}]"
        table_data.append([baseline_name, baseline_time_str, baseline_std_str, "(baseline)", "", "", "", baseline_ci_str])

        # --- Comparison Rows ---
        for item in comparisons:
            comp_data = item["data"]
            comp_name = item["name"]
            comp_mean = comp_data["stats"]["mean"]
            comp_std = comp_data["stats"].get("std", 0)
            comp_timings = comp_data["timings"]
            comp_n = len(comp_timings)

            _, p_value = ttest_ind(baseline_timings, comp_timings, equal_var=False)
            percent_change = ((comp_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0

            if comp_n > 1 and comp_std > 0:
                sem = comp_std / (comp_n ** 0.5)
                ci_margin = stats.t.ppf((1 + confidence_level) / 2, comp_n - 1) * sem
                comp_ci_lower = comp_mean - ci_margin
                comp_ci_upper = comp_mean + ci_margin
            else:
                comp_ci_lower, comp_ci_upper = comp_mean, comp_mean

            signif = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            time_str, _ = format_time(comp_mean)
            std_str = format_time(comp_std)[0]
            delta_str = f"{percent_change:+.2f}%" if baseline_mean > 0 else "n/a"
            factor_str = "n/a" if baseline_mean <= 0 or comp_mean <= 0 else (
                f"{comp_mean / baseline_mean:.2f}x slower" if comp_mean >= baseline_mean else
                f"{baseline_mean / comp_mean:.2f}x faster"
            )
            pval_str = f"{p_value:.3f}"
            ci_str = f"[{format_time(comp_ci_lower)[0]}, {format_time(comp_ci_upper)[0]}]"

            table_data.append([comp_name, time_str, std_str, delta_str, factor_str, pval_str, signif, ci_str])

        # --- Print Table ---
        table = tabulate(table_data, headers=table_headers, tablefmt="plain", numalign="right", stralign="left")
        separator = "-" * max(len(line) for line in table.split("\n"))
        self._logger.info(separator, separator=self._separator)
        for line in table.split("\n"):
            self._logger.info(line, separator=self._separator)
        self._logger.info(separator, separator=self._separator)

        # --- Summary ---
        fastest = min(comparisons + [baseline], key=lambda x: x["data"]["stats"]["mean"])
        slowest = max(comparisons + [baseline], key=lambda x: x["data"]["stats"]["mean"])
        self._logger.info(
            f"Summary: Fastest: {fastest['name']} ({format_time(fastest['data']['stats']['mean'])[0]}), "
            f"Slowest: {slowest['name']} ({format_time(slowest['data']['stats']['mean'])[0]})",
            separator=self._separator
        )

        # --- CSV Export Option ---
        export_csv = self._user_input_helper.get_user_input(
            "Export results to CSV? (y/n): ", expected_response_type=str,
            validator_func=lambda x: (x.lower() in ["y", "n"], "Enter 'y' or 'n'.")
        )
        if export_csv.lower() == "y":
            csv_path = self._category_dir / f"comparison_{baseline_name}_results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(table_headers)
                writer.writerows(table_data)
            self._logger.info(f"Results exported to {csv_path}", separator=self._separator)
