import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Union

from ..logging import HoornLogger


class CommandHelper:
	"""
	Helper class meant to streamline the execution of a command in command prompt. Enjoy.
	"""

	def __init__(self, hoorn_logger: HoornLogger, module_separator: str = "Commands"):
		"""
		Initializes the command helper with the provided HoornLogger and module separator.
        :param hoorn_logger: Logger instance.
        :param module_separator: String to separate module names in log messages.
        Below the separator root configured in the logger.
        """

		self._logger: HoornLogger = hoorn_logger
		self._module_separator = module_separator

	def _format_error(self, stderr: str) -> str:
		formatted = "Error executing command:\n"

		for line in stderr.split('\n'):
			formatted += f"  {line}\n"

		return formatted

	def execute_command(self, command: list, output_override: bool = False) -> subprocess.CompletedProcess:
		"""
		Executes a given command.
		Prints errors in all cases.
		"""
		if output_override:
			self._logger.info(f"Executing command: {command}", separator=self._module_separator)
			self._logger.info(f"Stringified: {' '.join(command)}", separator=self._module_separator)

		result = subprocess.run(command, capture_output=True)
		if result.returncode != 0:
			error_message = self._format_error(result.stderr.decode('utf-8'))
			self._logger.error(error_message, separator=self._module_separator)
			self._logger.info(f"Command causing error: {command}", separator=self._module_separator)

		return result

	def execute_command_v2(self, executable: Union[Path, str], command: list, shell: bool, hide_console: bool = True, keep_open: bool = False) -> None:
		"""Use this if `execute_command` does not work."""

		self._logger.debug(f"Executing {' '.join(command)} with executable {executable}", separator=self._module_separator)

		bat_file_path = Path(__file__).parent.joinpath("temp.bat")

		with open(bat_file_path, 'w') as bat_file:
			bat_file.write("@echo off\n")
			bat_file.write(f'"{executable}" {" ".join(command)}\n')

			if not keep_open:
				bat_file.write(f'exit\n')
			else: bat_file.write(f'pause\n')

		print(bat_file_path)

		if hide_console:
			subprocess.run(['start', '/b', os.environ["COMSPEC"], '/c', f"{bat_file_path}"], shell=shell)
			return

		subprocess.run(['start', os.environ["COMSPEC"], '/k', f"{bat_file_path}"], shell=shell)

	async def execute_command_v2_async(self, executable: Path, command: list, hide_console: bool = True, keep_open: bool = False) -> None:
		"""Use this if `execute_command` does not work. Async version."""

		self._logger.debug(f"Executing {' '.join(command)} with executable {executable}")

		bat_file_path = Path(__file__).parent.joinpath("temp.bat")

		executable_path = shutil.which(executable.name if type(executable) == Path else executable)

		with open(bat_file_path, 'w') as bat_file:
			bat_file.write(f'"{executable_path}" {" ".join(command)}\n')

			if not keep_open:
				bat_file.write(f'exit\n')
			else: bat_file.write(f'pause\nexit\n')

		print(bat_file_path)

		if hide_console:
			proc = await asyncio.create_subprocess_exec(
				os.environ["COMSPEC"],
				'/k',
				str(bat_file_path),
				shell=False
			)
			await proc.wait()
			return

		proc = await asyncio.create_subprocess_exec(
			os.environ["COMSPEC"],
			'/k',
			str(bat_file_path),
			shell=False
		)
		await proc.wait()
		return

	def open_python_module_with_custom_interpreter(self, interpreter_path: Path, working_directory: Path, module_name: str, args: list[str]):
		"""
		Opens a python module with a custom interpreter.
		"""

		self._logger.debug(f"Opening module {module_name} with interpreter {interpreter_path}", separator=self._module_separator)

		bat_file_path = Path(__file__).parent.joinpath("temp.bat")

		with open(bat_file_path, 'w') as bat_file:
			bat_file.write(f'cd "{working_directory}"\n')
			bat_file.write(f'"{interpreter_path}" -m {module_name} {" ".join(args)}\n')
			bat_file.write(f'pause\n')

		print(bat_file_path)
		subprocess.run(['start', os.environ["COMSPEC"], '/k', f"{bat_file_path}"], shell=True)
