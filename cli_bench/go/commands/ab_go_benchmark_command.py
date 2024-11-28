from abc import abstractmethod
from pathlib import Path
from typing import List, Union

from .go_command_context import GoCommandContext
from ...benchmark_command_interface import IBenchmarkCommand
from ...common.py_common.command_handling import CommandHelper
from ...common.py_common.handlers import FileHandler
from ...common.py_common.logging import HoornLogger
from ...common.py_common.user_input.user_input_helper import UserInputHelper
from ...utils.command_tools import CommandTools


class AbGoBenchmarkCommand(IBenchmarkCommand):
	"""
	Abstract Go benchmark command.
	Implements the IBenchmarkCommand interface and provides a basic structure for Go benchmark commands.
	"""

	def __init__(self, logger: HoornLogger, command_tools: CommandTools, command_context: GoCommandContext, is_child: bool = False):
		if not is_child:
			raise ValueError("AbGoBenchmarkCommand must be instantiated as a child command.")

		self._file_handler: FileHandler = command_tools.get_file_handler()
		self._command_handler: CommandHelper = command_tools.get_command_handler()
		self._user_input_helper: UserInputHelper = command_tools.get_user_input_helper()
		self._command_context: GoCommandContext = command_context

		super().__init__(logger, is_child=True)

	@abstractmethod
	def run(self) -> None:
		raise NotImplementedError("Subclasses must implement the run method.")

	def _get_benchmark_executable(self, prompt_message: str = "Enter the number of the benchmark executable:") -> Path | None:
		"""Gets the benchmark executable chosen by the user."""
		exes: List[Path] = self._file_handler.get_children_paths(self._command_context.benchmark_results_path, extension=".exe")
		if not exes:
			print("No benchmark executables found.")
			return None

		altered_prompt_message = ""

		for i, exe in enumerate(exes):
			altered_prompt_message += f"{i}. {exe.name}\n"

		altered_prompt_message += prompt_message

		choice = self._user_input_helper.get_user_input(altered_prompt_message, expected_response_type=int, validator_func=lambda x: self._validate_choice(x, len(exes)))
		if choice is None:  # Handle invalid input gracefully
			return None
		return exes[choice]

	def _validate_choice(self, choice: int, max_choice: int) -> [bool, str]:
		"""Validates the user's choice against the available options."""
		if not (0 <= choice < max_choice):
			return False, f"Invalid choice. Please enter a number between 0 and {max_choice - 1}."
		return True, ""

	def _resolve_benchmark_path(self, filename: str, extension: str = ".txt") -> Path:
		"""Resolves a path within the benchmark results directory."""
		return self._command_context.benchmark_results_path.joinpath(f"{filename}{extension}").resolve()

	async def _execute_go_command_async(self, commands: List[str], hide_console: bool = False, keep_open: bool = False, binary_override: Union[str, None] = None):
		await self._command_handler.execute_command_v2_async("go" if binary_override is None else binary_override, commands, hide_console=hide_console, keep_open=keep_open)

	def _execute_go_command(self, commands: List[str], shell: bool = True, hide_console: bool = False, keep_open: bool = False):
		self._command_handler.execute_command_v2("go", commands, shell=shell, hide_console=hide_console, keep_open=keep_open)
