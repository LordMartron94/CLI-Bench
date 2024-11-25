import asyncio
from pathlib import Path
from typing import List

from ...benchmark_command_interface import IBenchmarkCommand
from ...common.py_common.command_handling import CommandHelper
from ...common.py_common.handlers import FileHandler
from ...common.py_common.logging import HoornLogger
from .go_command_context import GoCommandContext


class BenchstatCommand(IBenchmarkCommand):
	def __init__(self, logger: HoornLogger, file_handler: FileHandler, command_handler: CommandHelper, command_context: GoCommandContext):
		self._file_helper: FileHandler = file_handler
		self._command_handler: CommandHelper = command_handler
		self._command_context: GoCommandContext = command_context

		super().__init__(logger, is_child=True)

	def run(self) -> None:
		async def run_benchstat_compare():
			results: List[Path] = self._file_helper.get_children_paths(self._command_context.benchmark_results_path, extension=".txt")

			if len(results) == 0:
				print("No benchmark results found in the results directory.")
				return

			results.sort(key=lambda x: x.name)
			print("Available benchmark results:")
			for i, result in enumerate(results):
				print(f"{i}. {result.name}")

			choices: List[int] = list(map(int, input("Enter the numbers of the benchmark results to compare (separated by spaces): ").split()))

			if len(choices) < 2:
				print("Please choose at least two benchmark results to compare.")
				return

			for i in range(len(choices)):
				if choices[i] < 0 or choices[i] >= len(results):
					print(f"Invalid choice for result {i+1}. Please try again.")
					return

			results: List[Path] = [results[i] for i in choices]

			with open(self._command_context.benchmark_interpretations_path.joinpath(f"{results[0].stem}-{results[-1].stem}.txt"), "w+") as tmpfile:
				temp_file_path = tmpfile.name

			command = [f"O=\"{results[0].resolve()}\""]

			for i in range(1, len(results)):
				command.append(f"N_{i}=\"{results[i].resolve()}\"")

			command.append(f"> \"{temp_file_path}\"")

			await self._command_handler.execute_command_v2_async("benchstat", command, hide_console=False, keep_open=False)

			print(f"Benchstat comparison (cleaned) written to: {temp_file_path}")

		asyncio.run(run_benchstat_compare())