import csv
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, shapiro, mannwhitneyu
from tabulate import tabulate

from ..benchmark_command_interface import IBenchmarkCommand
from ..common.py_common.logging import HoornLogger
from ..common.py_common.user_input.user_input_helper import UserInputHelper


def _fmt_time(seconds: float) -> Tuple[str, float]:
    if seconds is None or not math.isfinite(seconds):
        return "n/a", 1.0
    if seconds >= 1:
        return f"{seconds:.2f}s", 1
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.2f}ms", 1e3
    else:
        return f"{seconds * 1e6:.2f}µs", 1e6


def _fmt_bytes_to_mb(x: int | float | None) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "n/a"
    return f"{float(x) / (1024 * 1024):.1f} MB"


def _fmt_int(x: int | float | None) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "n/a"
    return str(int(x))


def _cohens_d(a: List[float] | np.ndarray, b: List[float] | np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = ((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / math.sqrt(pooled))


def _bootstrap_speedup_ci(
        baseline: List[float] | np.ndarray,
        comp: List[float] | np.ndarray,
        *,
        iters: int = 5000,
        alpha: float = 0.05,
        seed: int = 42,
) -> tuple[float, float]:
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(comp, dtype=float)
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, na, size=(iters, na))
    idx_b = rng.integers(0, nb, size=(iters, nb))
    means_a = a[idx_a].mean(axis=1)
    means_b = b[idx_b].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = means_a / means_b
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(ratios, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(ratios, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


# noinspection t
def _extract_mem_fields(doc: Dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    m = doc.get("memory") or {}
    rss_delta = None
    if isinstance(m.get("process"), dict):
        rss_delta = m["process"].get("rss_delta")
    if rss_delta is None:
        rss_delta = m.get("rss_delta")
    py_peak_op = None
    py_netnew_op = None
    allocs_op = None
    php = m.get("python_heap_per_op")
    if isinstance(php, dict):
        summ = php.get("summary", {}) or {}
        peak = summ.get("peak_bytes", {}) or {}
        netn = summ.get("net_new_bytes", {}) or {}
        alcs = summ.get("alloc_count", {}) or {}
        py_peak_op = peak.get("median") or peak.get("p95") or peak.get("max")
        py_netnew_op = netn.get("median") or netn.get("p95") or netn.get("max")
        allocs_op = alcs.get("median") or alcs.get("p95") or alcs.get("max")
    if py_peak_op is None:
        py_peak_op = m.get("tracemalloc_peak")
    rss_peak_op = None
    if isinstance(m.get("rss_peak_summary"), dict):
        rss_peak_op = m["rss_peak_summary"].get("median") or m["rss_peak_summary"].get("p95") or m["rss_peak_summary"].get("max")
    if rss_peak_op is None and isinstance(m.get("rss_peak_samples"), list) and m["rss_peak_samples"]:
        arr = np.asarray(m["rss_peak_samples"], dtype=float)
        rss_peak_op = float(np.median(arr))
    try:
        py_peak_op = int(py_peak_op) if py_peak_op is not None else None
    except Exception:
        py_peak_op = None
    try:
        py_netnew_op = int(py_netnew_op) if py_netnew_op is not None else None
    except Exception:
        py_netnew_op = None
    try:
        allocs_op = int(allocs_op) if allocs_op is not None else None
    except Exception:
        allocs_op = None
    try:
        rss_peak_op = int(rss_peak_op) if rss_peak_op is not None else None
    except Exception:
        rss_peak_op = None
    try:
        rss_delta = int(rss_delta) if rss_delta is not None else None
    except Exception:
        rss_delta = None
    return py_peak_op, py_netnew_op, allocs_op, rss_peak_op, rss_delta


class PythonCompareCommand(IBenchmarkCommand):
    def __init__(self, logger: HoornLogger, category_to_compare_dir: Path, *, confidence_level: float = 0.95):
        super().__init__(logger, is_child=True)
        self._separator = "Benchmarking.Python"
        self._category_dir = category_to_compare_dir
        self._user_input_helper = UserInputHelper(logger)
        self._confidence_level = confidence_level
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def run(self) -> None:
        self._logger.info(
            f"Analyzing benchmarks in category: '{self._category_dir.name}'",
            separator=self._separator,
        )
        result_files = sorted([p for p in self._category_dir.glob("*.json")])
        if len(result_files) < 2:
            self._logger.error(
                f"Need at least 2 results in '{self._category_dir.name}' to compare.",
                separator=self._separator,
            )
            return
        selected_files = self._get_user_selections(result_files)
        if len(selected_files) < 2:
            self._logger.warning(
                "Please select at least two benchmarks to compare.",
                separator=self._separator,
            )
            return
        all_data = []
        for file in selected_files:
            with open(file, "r", encoding="utf-8") as f:
                all_data.append({"data": json.load(f), "name": file.name})
        baseline = all_data[0]
        comparisons = all_data[1:]
        self._print_header(baseline["data"])
        self._compare_and_print(baseline, comparisons)

    def _get_user_selections(self, files: List[Path]) -> List[Path]:
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
        chosen_str = self._user_input_helper.get_user_input(
            prompt, expected_response_type=str, validator_func=validator
        )
        chosen_indices = [int(i) for i in chosen_str.strip().split()]
        return [files[i] for i in chosen_indices]

    def _print_header(self, data: Dict[str, Any]) -> None:
        env = data.get("environment", {}) or {}
        cfg = data.get("timing_config", {}) or {}
        meta = data.get("metadata", {}) or {}
        env_lines = [
            f"Python {env.get('python','?')} | {env.get('platform','?')} | CPU: {env.get('cpu','?')}",
            f"CPUs: logical={env.get('logical_cpus','?')}, physical={env.get('physical_cpus','?')}",
            f"Packages: numpy={env.get('numpy')}, scipy={env.get('scipy')}, librosa={env.get('librosa')}, torch={env.get('torch')}",
            f"BLAS threads: {env.get('blas_threads')}",
            f"Git: {env.get('git')}",
        ]
        cfg_lines = [
            f"Repeat={cfg.get('repeat','?')}, NumberMode={cfg.get('number_mode','?')}, Loops/Rep={cfg.get('loops_per_repeat','?')}",
            f"Threads={cfg.get('threads','?')}, PinAffinity={cfg.get('pin_affinity','?')}, AutorangeNote={cfg.get('autorange_note','')}",
            f"Category={meta.get('category','?')}, Timestamp={meta.get('timestamp_utc','?')}",
        ]
        self._logger.info("\n".join(env_lines), separator=self._separator)
        self._logger.info("\n".join(cfg_lines), separator=self._separator)

    # noinspection t
    def _compare_and_print(self, baseline: Dict[str, Any], comparisons: List[Dict[str, Any]]) -> None:
        baseline_data = baseline["data"]
        baseline_name = baseline["name"]
        baseline_mean = baseline_data["stats"]["mean"]
        baseline_std = baseline_data["stats"].get("stdev", 0.0)
        baseline_timings = baseline_data["timings"]
        baseline_n = len(baseline_timings)
        baseline_rsd = baseline_data["stats"].get("rsd_percent", None)
        cl = float(self._confidence_level)
        if baseline_n > 1 and baseline_std > 0:
            sem = baseline_std / (baseline_n ** 0.5)
            ci_margin = stats.t.ppf((1 + cl) / 2, baseline_n - 1) * sem
            baseline_ci_lower = baseline_mean - ci_margin
            baseline_ci_upper = baseline_mean + ci_margin
        else:
            baseline_ci_lower, baseline_ci_upper = baseline_mean, baseline_mean
        headers = [
            "Benchmark", "N", "Time", "Std Dev", "RSD",
            "Delta", "Factor (×) + Orders + CI",
            "P-value", "Signif", "95% CI (mean)", "EffSz(d)",
            "Py Peak/op", "Py NetNew/op", "Allocs/op", "RSS Peak/op", "RSS Δ (probe)"
        ]
        b_time, _ = _fmt_time(baseline_mean)
        b_std = _fmt_time(baseline_std)[0]
        b_ci = f"[{_fmt_time(baseline_ci_lower)[0]}, {_fmt_time(baseline_ci_upper)[0]}]"
        b_rsd = f"{baseline_rsd:.2f}%" if isinstance(baseline_rsd, (float, int)) and math.isfinite(float(baseline_rsd)) else "n/a"
        b_py_peak, b_py_netnew, b_allocs, b_rss_peak, b_rss_delta = _extract_mem_fields(baseline_data)
        table_rows = [[
            baseline_name, baseline_n, b_time, b_std, b_rsd,
            "(baseline)", "", "", "", b_ci, "",
            _fmt_bytes_to_mb(b_py_peak), _fmt_bytes_to_mb(b_py_netnew), _fmt_int(b_allocs),
            _fmt_bytes_to_mb(b_rss_peak), _fmt_bytes_to_mb(b_rss_delta)
        ]]
        for item in comparisons:
            comp_data = item["data"]
            comp_name = item["name"]
            comp_mean = comp_data["stats"]["mean"]
            comp_std = comp_data["stats"].get("stdev", 0.0)
            comp_timings = comp_data["timings"]
            comp_n = len(comp_timings)
            comp_rsd = comp_data["stats"].get("rsd_percent", None)
            use_mw = False
            if comp_n >= 3 and baseline_n >= 3:
                try:
                    p_shapiro_b = shapiro(baseline_timings).pvalue
                    p_shapiro_c = shapiro(comp_timings).pvalue
                    use_mw = (p_shapiro_b < 0.05 or p_shapiro_c < 0.05)
                except Exception:
                    use_mw = False
            if use_mw:
                try:
                    _, p_value = mannwhitneyu(baseline_timings, comp_timings, alternative="two-sided")
                except Exception:
                    _, p_value = ttest_ind(baseline_timings, comp_timings, equal_var=False)
            else:
                _, p_value = ttest_ind(baseline_timings, comp_timings, equal_var=False)
            if comp_n > 1 and comp_std > 0:
                sem = comp_std / (comp_n ** 0.5)
                ci_margin = stats.t.ppf((1 + cl) / 2, comp_n - 1) * sem
                comp_ci_lower = comp_mean - ci_margin
                comp_ci_upper = comp_mean + ci_margin
            else:
                comp_ci_lower, comp_ci_upper = comp_mean, comp_mean
            percent_change = ((comp_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            delta_str = f"{percent_change:+.2f}%"
            factor_desc = "n/a"
            if baseline_mean > 0 and comp_mean > 0 and math.isfinite(comp_mean):
                if comp_mean >= baseline_mean:
                    slow = comp_mean / baseline_mean
                    orders_slow = math.log10(slow) if slow > 0 and math.isfinite(slow) else float("inf")
                    factor_desc = f"{slow:.2f}× slower"
                    if math.isfinite(orders_slow):
                        factor_desc += f"; {orders_slow:.2f} orders"
                else:
                    speedup = baseline_mean / comp_mean
                    orders_fast = math.log10(speedup) if speedup > 0 and math.isfinite(speedup) else float("inf")
                    factor_desc = f"{speedup:.2f}× faster"
                    if math.isfinite(orders_fast):
                        factor_desc += f"; {orders_fast:.2f} orders"
                ci_lo, ci_hi = _bootstrap_speedup_ci(baseline_timings, comp_timings, iters=4000, alpha=1 - cl)
                if math.isfinite(ci_lo) and math.isfinite(ci_hi):
                    factor_desc += f" [CI: {ci_lo:.2f}–{ci_hi:.2f}]"
            d = _cohens_d(baseline_timings, comp_timings)
            eff = f"{d:.2f}" if math.isfinite(d) else "n/a"
            c_py_peak, c_py_netnew, c_allocs, c_rss_peak, c_rss_delta = _extract_mem_fields(comp_data)
            time_str, _ = _fmt_time(comp_mean)
            std_str = _fmt_time(comp_std)[0]
            rsd_str = f"{comp_rsd:.2f}%" if isinstance(comp_rsd, (float, int)) and math.isfinite(float(comp_rsd)) else "n/a"
            if isinstance(comp_rsd, (float, int)) and math.isfinite(float(comp_rsd)) and float(comp_rsd) > 10.0:
                rsd_str += " ⚠︎"
            ci_str = f"[{_fmt_time(comp_ci_lower)[0]}, {_fmt_time(comp_ci_upper)[0]}]"
            signif = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            table_rows.append([
                comp_name, comp_n, time_str, std_str, rsd_str,
                delta_str, factor_desc, f"{p_value:.3f}", signif, ci_str, eff,
                _fmt_bytes_to_mb(c_py_peak), _fmt_bytes_to_mb(c_py_netnew), _fmt_int(c_allocs),
                _fmt_bytes_to_mb(c_rss_peak), _fmt_bytes_to_mb(c_rss_delta)
            ])
        table = tabulate(table_rows, headers=headers, tablefmt="plain", numalign="right", stralign="left")
        separator = "-" * max(len(line) for line in table.split("\n"))
        self._logger.info(separator, separator=self._separator)
        for line in table.split("\n"):
            self._logger.info(line, separator=self._separator)
        self._logger.info(separator, separator=self._separator)
        fastest = min(comparisons + [baseline], key=lambda x: x["data"]["stats"]["mean"])
        slowest = max(comparisons + [baseline], key=lambda x: x["data"]["stats"]["mean"])
        self._logger.info(
            f"Summary: Fastest: {fastest['name']} ({_fmt_time(fastest['data']['stats']['mean'])[0]}), "
            f"Slowest: {slowest['name']} ({_fmt_time(slowest['data']['stats']['mean'])[0]})",
            separator=self._separator,
        )
        export_csv = self._user_input_helper.get_user_input(
            "Export results to CSV? (y/n): ",
            expected_response_type=str,
            validator_func=lambda x: (x.lower() in ["y", "n"], "Enter 'y' or 'n'."),
        )
        if export_csv.lower() == "y":
            csv_path = self._category_dir / f"comparison_{baseline_name}_results.csv"
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for row in table_rows:
                    writer.writerow([cell if isinstance(cell, str) else str(cell) for cell in row])
            self._logger.info(f"Results exported to {csv_path}", separator=self._separator)
