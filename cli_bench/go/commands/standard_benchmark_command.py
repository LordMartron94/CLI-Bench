import asyncio
import os
import time
from pathlib import Path
from typing import List

from ...common.py_common.command_handling import CommandHelper
from ...common.py_common.handlers import FileHandler
from ...common.py_common.logging import HoornLogger
from ...benchmark_command_interface import IBenchmarkCommand
from .go_command_context import GoCommandContext
from ...common.py_common.user_input.user_input_helper import UserInputHelper


class StandardBenchmarkCommand(IBenchmarkCommand):
	def __init__(self, logger: HoornLogger, file_handler: FileHandler, command_handler: CommandHelper, user_input_helper: UserInputHelper, command_context: GoCommandContext):
		self._file_helper: FileHandler = file_handler
		self._command_handler: CommandHelper = command_handler
		self._user_input_helper: UserInputHelper = user_input_helper
		self._command_context: GoCommandContext = command_context

		super().__init__(logger, is_child=True)

	def _validate_files(self, go_files: List[Path]) -> bool:
		if len(go_files) == 0:
			print("No benchmark files found in the benchmarks directory.")
			return False

		return True

	def run(self) -> None:
		async def run_benchmark():
			go_files = self._file_helper.get_children_paths(self._command_context.benchmarks_path, extension=".go")

			if not self._validate_files(go_files):
				return

			choice = self._get_benchmark_choice(go_files)
			output_name = self._user_input_helper.get_user_input("Enter the output name for the benchmark results:", expected_response_type=str, validator_func=lambda x: True)

			benchmark_file = self._command_context.benchmarks_path.joinpath(choice.name)

			desired_count = self._user_input_helper.get_user_input("Enter the number of times to run the benchmark (leave empty for 10):", expected_response_type=str, validator_func=lambda x: True)
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
