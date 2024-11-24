from cli_bench.common.py_common.logging import HoornLogger, DefaultHoornLogOutput, LogType
from cli_bench.go.commands.go_command_context import GoCommandContextFactory
from cli_bench.go.go_cli import GolangCLI

def start_golang_cli(module_root: str, benchmarks_path: str, benchmark_results_path: str, benchmark_interpretations_path: str):
	logger = HoornLogger(min_level=LogType.DEBUG, outputs=[DefaultHoornLogOutput()])

	cli = GolangCLI(logger, GoCommandContextFactory.create(module_root, benchmarks_path, benchmark_results_path, benchmark_interpretations_path))
	cli.start()
