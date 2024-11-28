from .ab_go_benchmark_command import AbGoBenchmarkCommand
from ...common.py_common.logging import HoornLogger
from .go_command_context import GoCommandContext
from ...utils.command_tools import CommandTools


class PprofWebCommand(AbGoBenchmarkCommand):
	def __init__(self, logger: HoornLogger, command_tools: CommandTools, command_context: GoCommandContext):
		super().__init__(logger, command_tools, command_context, is_child=True)

	def run(self) -> None:
		executable = self._get_benchmark_executable()
		if executable is None:
			return

		cpu_result = self._resolve_benchmark_path(executable.stem, ".cpu.prof")

		commands = [
			"tool", "pprof", "-http=\":8080\"", str(executable), str(cpu_result)
		]

		self._execute_go_command(commands, hide_console=False, keep_open=True, shell=True)
