from pathlib import Path

import pydantic


class GoCommandContext(pydantic.BaseModel):
	module_root: Path
	benchmarks_path: Path
	benchmark_results_path: Path
	benchmark_interpretations_path: Path

class GoCommandContextFactory:
	@staticmethod
	def create(module_root: str, benchmarks_path: str, benchmark_results_path: str, benchmark_interpretations_path: str) -> "GoCommandContext":
		return GoCommandContext(
            module_root=Path(module_root),
            benchmarks_path=Path(benchmarks_path),
            benchmark_results_path=Path(benchmark_results_path),
            benchmark_interpretations_path=Path(benchmark_interpretations_path)
        )
