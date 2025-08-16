import gc
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import statistics
import threading
import time
import timeit
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import psutil

from .stable_timing import stable_timing
from ..benchmark_command_interface import IBenchmarkCommand
from ..common.py_common.logging import HoornLogger
from ..common.py_common.user_input.user_input_helper import UserInputHelper

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def _pkg_version(name: str) -> Optional[str]:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _perc(vals, q):
    return float(np.percentile(vals, q)) if vals else float("nan")


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
            number: int | str = 1000,
            *,
            threads: int = 1,
            pin_affinity: bool = False,
            autorange_target_comment: str = "timeit default (~0.2s)",

            # --- memory/alloc profiling toggles ---
            measure_memory: bool = True,
            mem_probe_runs: int = 5,
            alloc_hotspots: bool = True,
            alloc_hotspots_top_n: int = 20,
            alloc_group_by: str = "lineno",
            capture_gpu_memory: bool = True,
    ) -> None:
        super().__init__(logger, is_child=True)
        self._separator = "Benchmarking.Python"
        self._benchmark_functions = benchmark_functions
        self._category_dir = category_dir
        self._repeat = repeat
        self._number = number
        self._threads = threads
        self._pin_affinity = pin_affinity
        self._autorange_target_comment = autorange_target_comment

        self._measure_memory = measure_memory
        self._mem_probe_runs = max(1, mem_probe_runs)
        self._alloc_hotspots = alloc_hotspots
        self._alloc_hotspots_top_n = max(1, alloc_hotspots_top_n)
        self._alloc_group_by = alloc_group_by
        self._capture_gpu_memory = capture_gpu_memory

        self._user_input_helper = UserInputHelper(logger)
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def run(self) -> None:
        """Runs all benchmarks for the functions provided at initialization."""
        self._category_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(
            f"Running benchmarks for category: '{self._category_dir.name}'",
            separator=self._separator,
        )

        for func in self._benchmark_functions:
            raw_name = func.__name__
            sanitized_name = re.sub(r'[<>:"/\\|?*]', "_", raw_name)
            if not sanitized_name or sanitized_name == "_lambda_":
                self._logger.error(
                    "Cannot create a benchmark for an unnamed or lambda function. "
                    "Please wrap it in a named function.",
                    separator=self._separator,
                )
                continue

            output_path = self._category_dir / f"{sanitized_name}.json"
            self._execute_and_save_benchmark(func, sanitized_name, output_path)

    # noinspection t
    def _execute_and_save_benchmark(self, fn: Callable, name: str, output_path: Path) -> None:
        self._logger.info(f"Running benchmark for '{name}'...", separator=self._separator)

        # Warmup (lets JITs/caches settle; avoids polluting timing and mem probes)
        self._logger.trace(f"Performing warm-up run for '{name}'...", separator=self._separator)
        gc.collect()
        fn()

        timer = timeit.Timer(fn)
        if isinstance(self._number, str) and self._number == "auto":
            loops, _ = timer.autorange()
            loops_per_repeat = loops
        else:
            loops_per_repeat = int(self._number)

        self._logger.debug(f"Configuration: [repetitions={self._repeat}, loops={loops_per_repeat}]", separator=self._separator)

        # -------------------------
        # Timings with stable env
        # -------------------------
        with stable_timing(threads=self._threads, pin_affinity=self._pin_affinity):
            timings = timer.repeat(repeat=self._repeat, number=loops_per_repeat)

        timings_per_op = [t / loops_per_repeat for t in timings]

        # Robust time stats
        mean = statistics.mean(timings_per_op)
        stdev = statistics.stdev(timings_per_op) if len(timings_per_op) > 1 else 0.0
        median = statistics.median(timings_per_op)
        min_val, max_val = min(timings_per_op), max(timings_per_op)
        pcts = {q: float(np.percentile(timings_per_op, q)) for q in (50, 90, 95, 99)}
        iqr = float(np.percentile(timings_per_op, 75) - np.percentile(timings_per_op, 25))
        rsd = float((stdev / mean) * 100.0) if mean > 0 else float("nan")

        # -------------------------------------------------------
        # Optional memory/alloc probes (kept separate from timing)
        # -------------------------------------------------------
        def _rss_peak_while_running(callable_fn, interval_sec: float = 0.005) -> tuple[int, float]:
            """Run fn while sampling RSS; return (peak_rss_bytes, duration_sec)."""
            proc = psutil.Process()
            stop = threading.Event()
            peak_rss = {"v": proc.memory_info().rss}
            t0 = time.perf_counter()

            def _poll():
                while not stop.is_set():
                    try:
                        rss = proc.memory_info().rss
                        if rss > peak_rss["v"]:
                            peak_rss["v"] = rss
                    except Exception:
                        pass
                    time.sleep(interval_sec)

            th = threading.Thread(target=_poll, daemon=True)
            th.start()
            try:
                callable_fn()
            finally:
                stop.set()
                th.join()
            return peak_rss["v"], (time.perf_counter() - t0)

        peak_tracemalloc_samples: list[int] = []
        current_tracemalloc_samples: list[int] = []
        net_new_bytes_samples: list[int] = []
        alloc_count_samples: list[int] = []
        rss_peak_samples: list[int] = []
        hotspot_rows: list[dict] = []

        # Capture process RSS around the whole probe phase (not per-op)
        proc = psutil.Process()
        rss_before = int(proc.memory_info().rss)

        # Depth of traceback kept in tracemalloc; 10–25 is a good range
        traceback_limit = 15
        top_k_hotspots = 15

        for _ in range(max(1, self._mem_probe_runs)):
            gc.collect()
            tracemalloc.start(traceback_limit)
            try:
                # snapshot before
                snap_before = tracemalloc.take_snapshot()

                # run op with RSS sampling
                rss_peak, _dur = _rss_peak_while_running(fn)

                # snapshot after and peak/current
                snap_after = tracemalloc.take_snapshot()
                current_bytes, peak_bytes = tracemalloc.get_traced_memory()

                # net-new bytes & count (lower bound: objects that survived the op)
                diffs = snap_after.compare_to(snap_before, 'traceback')
                net_new_bytes = sum(max(0, d.size_diff) for d in diffs)
                alloc_count = sum(max(0, d.count_diff) for d in diffs)

                # record samples
                peak_tracemalloc_samples.append(int(peak_bytes))
                current_tracemalloc_samples.append(int(current_bytes))
                net_new_bytes_samples.append(int(net_new_bytes))
                alloc_count_samples.append(int(alloc_count))
                rss_peak_samples.append(int(rss_peak))

                diffs_sorted = sorted(
                    (d for d in diffs if d.size_diff > 0),
                    key=lambda d: (d.size_diff, d.count_diff),
                    reverse=True
                )[:top_k_hotspots]
                for d in diffs_sorted:
                    tb = d.traceback
                    frame = tb[-1] if tb else None
                    hotspot_rows.append({
                        "file": getattr(frame, "filename", None),
                        "line": getattr(frame, "lineno", None),
                        "trace": str(tb).splitlines()[-1] if tb else None,
                        "size_diff": int(d.size_diff),
                        "count_diff": int(d.count_diff),
                    })
            finally:
                tracemalloc.stop()

        rss_after = int(proc.memory_info().rss)
        rss_delta = max(0, rss_after - rss_before)

        def _summary(vals: list[int]) -> dict:
            if not vals:
                return {"median": None, "p95": None, "max": None}
            arr = np.asarray(vals, dtype=np.int64)
            return {
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "max": int(arr.max()),
            }

        mem = {
            "process": {
                "rss_before": rss_before,
                "rss_after": rss_after,
                "rss_delta": rss_delta,
            },
            "python_heap_per_op": {
                "peak_bytes_samples": peak_tracemalloc_samples,
                "current_bytes_samples": current_tracemalloc_samples,
                "net_new_bytes_samples": net_new_bytes_samples,
                "alloc_count_samples": alloc_count_samples,
                "summary": {
                    "peak_bytes": _summary(peak_tracemalloc_samples),
                    "net_new_bytes": _summary(net_new_bytes_samples),
                    "alloc_count": _summary(alloc_count_samples),
                },
                "notes": "tracemalloc tracks Python heap only; net_new is a lower bound (allocs freed inside the op are not counted).",
            },
            "rss_peak_samples": rss_peak_samples,
            "rss_peak_summary": _summary(rss_peak_samples),
            "hotspots_top": sorted(hotspot_rows, key=lambda r: (r["size_diff"], r["count_diff"]), reverse=True)[:top_k_hotspots],
            "gpu": {"peak_bytes": 0},
        }

        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "numpy": _pkg_version("numpy"),
            "scipy": _pkg_version("scipy"),
            "librosa": _pkg_version("librosa"),
            "torch": _pkg_version("torch"),
            "blas_threads": {k: os.getenv(k) for k in ["MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
            "git": os.popen("git rev-parse --short HEAD 2>NUL").read().strip() or None,
        }

        timing_cfg = {
            "repeat": self._repeat,
            "number_mode": "auto" if (isinstance(self._number, str) and self._number == "auto") else "fixed",
            "loops_per_repeat": loops_per_repeat,
            "threads": self._threads,
            "pin_affinity": self._pin_affinity,
            "autorange_note": self._autorange_target_comment,
        }

        results_data = {
            "environment": env,
            "metadata": {
                "name": name,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "category": self._category_dir.name,
            },
            "timing_config": timing_cfg,
            "stats": {
                "unit": "seconds_per_op",
                "mean": mean,
                "stdev": stdev,
                "median": median,
                "min": min_val,
                "max": max_val,
                "pcts": pcts,
                "iqr": iqr,
                "rsd_percent": rsd,
                "samples": len(timings_per_op),
            },
            "memory": mem,
            "timings": timings_per_op,
        }

        self._logger.info(
            f"📊 Results for '{name}': Mean={mean * 1e6:.2f} ± {stdev * 1e6:.2f} µs/op | "
            f"Median={median * 1e6:.2f} µs | Range=[{min_val * 1e6:.2f}, {max_val * 1e6:.2f}] µs | RSD={rsd:.2f}%",
            separator=self._separator,
        )

        summary = mem["python_heap_per_op"]["summary"]
        median_net_new_bytes = summary["net_new_bytes"]["median"]
        median_peak_bytes = summary["peak_bytes"]["median"]
        median_allocs = summary["alloc_count"]["median"]

        net_new_str = f"{median_net_new_bytes / 1024:.2f} KB" if median_net_new_bytes is not None else "N/A"
        peak_str = f"{median_peak_bytes / 1024:.2f} KB" if median_peak_bytes is not None else "N/A"
        allocs_str = f"{int(median_allocs)}" if median_allocs is not None else "N/A"

        self._logger.info(
            f"🧠 Memory for '{name}': Net New={net_new_str}/op | Peak Heap={peak_str}/op | Allocs={allocs_str}/op",
            separator=self._separator,
        )

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=4)
        self._logger.info(
            f"Benchmark results for '{name}' saved to: {output_path}",
            separator=self._separator,
        )
