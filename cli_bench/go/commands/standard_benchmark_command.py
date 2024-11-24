import asyncio
import os
import time
from pathlib import Path
from typing import List

from cli_bench.common.py_common.command_handling import CommandHelper
from cli_bench.common.py_common.handlers import FileHandler
from cli_bench.common.py_common.logging import HoornLogger
from cli_bench.benchmark_command_interface import IBenchmarkCommand
from cli_bench.go.commands.go_command_context import GoCommandContext


class StandardBenchmarkCommand(IBenchmarkCommand):
	def __init__(self, logger: HoornLogger, file_handler: FileHandler, command_handler: CommandHelper, command_context: GoCommandContext):
		self._file_helper: FileHandler = file_handler
		self._command_handler: CommandHelper = command_handler
		self._command_context: GoCommandContext = command_context

		super().__init__(logger, is_child=True)

	def run(self) -> None:
		async def run_benchmark():
			go_files = self._file_helper.get_children_paths(self._command_context.benchmarks_path, extension=".go")

			if len(go_files) == 0:
				print("No benchmark files found in the benchmarks directory.")
				return

			choice = self._get_benchmark_choice(go_files)
			output_name = input("Enter the output name for the benchmark results: ")

			benchmark_file = self._command_context.benchmarks_path.joinpath(choice.name)

			desired_count = input("Enter the number of times to run the benchmark (leave empty for 10): ")
			desired_count = int(desired_count) if desired_count else 10

			commands = [
				"test",
				f"\"{benchmark_file.__str__()}\"",
				"-run=^$",
				f"-count={desired_count}",
				"-bench=.",
				"-benchmem",
				"-cpuprofile",
				f"\"{self._command_context.benchmark_results_path.joinpath(f'{output_name}.cpu.prof').resolve()}\"",
				"-memprofile",
				f"\"{self._command_context.benchmark_results_path.joinpath(f'{output_name}.mem.prof').resolve()}\"",
				f"-o=\"{self._command_context.benchmark_results_path.joinpath(f'{output_name}.exe').resolve()}\"",
				f"> \"{self._command_context.benchmark_results_path.joinpath(f'{output_name}.txt').resolve()}\""
			]

			commands_2 = [
				"tool",
				"pprof",
				f"\"{self._command_context.benchmark_results_path.joinpath(f'{output_name}.exe').resolve()}\"",
				f"\"{self._command_context.benchmark_results_path.joinpath(f'{output_name}.cpu.prof')}\""
			]

			os.chdir(self._command_context.module_root)

			await self._command_handler.execute_command_v2_async("go", commands, hide_console=False, keep_open=False)
			await self._command_handler.execute_command_v2_async("go", commands_2, hide_console=False, keep_open=False)

		asyncio.run(run_benchmark())

	def _get_benchmark_choice(self, go_files: List[Path]) -> Path:
		for i, file in enumerate(go_files):
			print(f"{i}. {file.name}")

		choice = int(input("Enter the number of the benchmark file: "))

		if choice < 0 or choice >= len(go_files):
			print("Invalid choice. Please try again.")
			time.sleep(0.5)
			return self._get_benchmark_choice(go_files)

		return go_files[choice]