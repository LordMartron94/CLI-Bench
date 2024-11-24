from pathlib import Path
from typing import List

from cli_bench.benchmark_command_interface import IBenchmarkCommand
from cli_bench.common.py_common.command_handling import CommandHelper
from cli_bench.common.py_common.handlers import FileHandler
from cli_bench.common.py_common.logging import HoornLogger
from cli_bench.go.commands.go_command_context import GoCommandContext


class PprofInteractableCommand(IBenchmarkCommand):
	def __init__(self, logger: HoornLogger, file_handler: FileHandler, command_handler: CommandHelper, command_context: GoCommandContext):
		self._file_helper: FileHandler = file_handler
		self._command_handler: CommandHelper = command_handler
		self._command_context: GoCommandContext = command_context

		super().__init__(logger, is_child=True)

	def run(self) -> None:
		exes: List[Path] = self._file_helper.get_children_paths(self._command_context.benchmark_results_path, extension=".exe")
		if len(exes) == 0:
			print("No benchmark executables found in the benchmark results directory.")
			return

		for i, exe in enumerate(exes):
			print(f"{i}. {exe.name}")

		chosen_executable = int(input("Enter the number of the benchmark executable: "))
		if chosen_executable < 0 or chosen_executable >= len(exes):
			print("Invalid choice. Please try again.")
			return self.run()

		executable = exes[chosen_executable]
		cpu_result = self._command_context.benchmark_results_path.joinpath(f"{executable.stem}.cpu.prof")

		commands = [
			"tool",
			"pprof",
			f"\"{executable.resolve()}\"",
			f"\"{cpu_result.resolve()}\""
		]

		self._command_handler.execute_command_v2("go", commands, shell=True, hide_console=False, keep_open=True)