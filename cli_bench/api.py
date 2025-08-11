from .go.commands.go_command_context import GoCommandContextFactory
from .go.go_cli import GolangCLI
from ...md_common_python.py_common.logging import HoornLoggerBuilder, LogType


def start_golang_cli(module_root: str, benchmarks_path: str, benchmark_results_path: str, benchmark_interpretations_path: str):
	logger_builder: HoornLoggerBuilder = HoornLoggerBuilder(
		application_name_sanitized="MD.Benchmarker",
		max_separator_length=30
	)

	logger = logger_builder.build_console_output().get_logger(min_level=LogType.DEBUG)

	cli = GolangCLI(logger, GoCommandContextFactory.create(module_root, benchmarks_path, benchmark_results_path, benchmark_interpretations_path))
	cli.start()
