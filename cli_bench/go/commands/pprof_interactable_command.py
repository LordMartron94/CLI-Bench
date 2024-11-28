from pathlib import Path
from typing import List

from .ab_go_benchmark_command import AbGoBenchmarkCommand
from ...common.py_common.logging import HoornLogger
from .go_command_context import GoCommandContext
from ...utils.command_tools import CommandTools


class PprofInteractableCommand(AbGoBenchmarkCommand):
	def __init__(self, logger: HoornLogger, command_tools: CommandTools, command_context: GoCommandContext):
		super().__init__(logger, command_tools, command_context)

	def _validate_paths(self, exes: List[Path]) -> bool:
		if len(exes) == 0:
			print("No benchmark executables found in the benchmark results directory.")
			return False

		return True

	def run(self) -> None:
		executable = self._get_benchmark_executable()
		if executable is None:
			return

		cpu_result = self._resolve_benchmark_path(executable.stem, ".cpu.prof")

		commands = [
			"tool", "pprof", str(executable), str(cpu_result)
		]

		self._command_handler.execute_command_v2("go", commands, shell=True, hide_console=False, keep_open=True)
