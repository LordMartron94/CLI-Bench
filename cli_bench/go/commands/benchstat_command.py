import asyncio
from pathlib import Path
from typing import List

from .ab_go_benchmark_command import AbGoBenchmarkCommand
from ...common.py_common.logging import HoornLogger
from .go_command_context import GoCommandContext
from ...utils.command_tools import CommandTools


class BenchstatCommand(AbGoBenchmarkCommand):
	def __init__(self, logger: HoornLogger, command_tools: CommandTools, command_context: GoCommandContext):
		super().__init__(logger, command_tools, command_context, is_child=True)

	def run(self) -> None:
		async def run_benchstat_compare():
			results: List[Path] = self._file_handler.get_children_paths(self._command_context.benchmark_results_path, extension=".txt")

			if not results:
				print("No benchmark results found.")
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

			output_path = self._resolve_benchmark_path(f"{results[0].stem}-{results[-1].stem}")

			command = [f"O={results[0]}"]

			for i in range(1, len(results)):
				command.append(f"N_{i}={results[i]}")

			command.append(f"> {output_path}")

			await self._execute_go_command_async(command, binary_override="benchstat")

			print(f"Benchstat comparison written to: {output_path}")

		asyncio.run(run_benchstat_compare())
