import asyncio
import os
from pathlib import Path
from typing import List

from .ab_go_benchmark_command import AbGoBenchmarkCommand
from ...common.py_common.logging import HoornLogger
from .go_command_context import GoCommandContext
from ...utils.command_tools import CommandTools


class StandardBenchmarkCommand(AbGoBenchmarkCommand):
	def __init__(self, logger: HoornLogger, command_tools: CommandTools, command_context: GoCommandContext):
		super().__init__(logger, command_tools, command_context)

	def _validate_files(self, go_files: List[Path]) -> bool:
		if len(go_files) == 0:
			print("No benchmark files found in the benchmarks directory.")
			return False

		return True

	def run(self) -> None:
		async def run_benchmark():
			go_files = self._file_handler.get_children_paths(self._command_context.benchmarks_path, extension=".go")

			if not self._validate_files(go_files):
				return

			choice = self._get_benchmark_choice(go_files)
			output_name = self._user_input_helper.get_user_input("Enter the output name for the benchmark results:", expected_response_type=str, validator_func=lambda x: [True, ""])

			benchmark_file = self._command_context.benchmarks_path.joinpath(choice.name)

			cpu_prof_path = self._resolve_benchmark_path(output_name, ".cpu.prof")
			mem_prof_path = self._resolve_benchmark_path(output_name, ".mem.prof")
			exe_path = self._resolve_benchmark_path(output_name, ".exe")
			results_path = self._resolve_benchmark_path(output_name)

			desired_count = self._user_input_helper.get_user_input("Enter the number of times to run the benchmark (leave empty for 10):", expected_response_type=str, validator_func=lambda x: [True, ""])
			desired_count = int(desired_count) if desired_count else 10

			commands = [
				"test",
				str(benchmark_file),
				"-run=^$",
				f"-count={desired_count}",
				"-bench=.",
				"-benchmem",
				"-cpuprofile", str(cpu_prof_path),
				"-memprofile", str(mem_prof_path),
				f"-o={exe_path}",
				f"> {results_path}"
			]

			commands_2 = [
				"tool", "pprof", str(exe_path), str(cpu_prof_path)
			]

			os.chdir(self._command_context.module_root)

			await self._execute_go_command_async(commands)
			await self._execute_go_command_async(commands_2)

		asyncio.run(run_benchmark())

	def _get_benchmark_choice(self, go_files: List[Path]) -> Path:
		prompt: str = ""

		for i, file in enumerate(go_files):
			prompt += f"{i}. {file.name}\n"

		prompt += "Enter the number of the benchmark file: "
		choice = self._user_input_helper.get_user_input(prompt, expected_response_type=int, validator_func=lambda chosen: self._validate_choice(chosen, len(go_files)))

		return go_files[choice]

	def _validate_choice(self, choice: int, len_go_files: int) -> [bool, str]:
		if choice < 0 or choice >= len_go_files:
			return False, f"Invalid choice. Please enter a number between 0 and {len_go_files - 1}."

		return True, ""
