import os

from ..common.py_common.cli_framework import CommandLineInterface
from ..common.py_common.logging import HoornLogger
from .commands.benchstat_command import BenchstatCommand
from .commands.go_command_context import GoCommandContext
from .commands.pprof_interactable_command import PprofInteractableCommand
from .commands.pprof_web_command import PprofWebCommand
from .commands.standard_benchmark_command import StandardBenchmarkCommand
from ..utils.command_tools import CommandTools


class GolangCLI:
	def __init__(self, logger: HoornLogger, command_context: GoCommandContext):
		self._cli_interface = CommandLineInterface(logger)
		self._logger: HoornLogger = logger

		_command_tools: CommandTools = CommandTools(self._logger)

		# Initialize directories to avoid "system cannot find the path" errors.
		os.makedirs(command_context.benchmark_results_path, exist_ok=True)
		os.makedirs(command_context.benchmark_interpretations_path, exist_ok=True)

		self._commands_to_add = {
			0: {
				"keys": ["benchmark", "bm"],
				"description": "Starts the benchmarking suite.",
				"command": StandardBenchmarkCommand(self._logger, _command_tools, command_context).run
			},
			1: {
				"keys": ["benchstat-compare", "bsc"],
                "description": "Compares two benchmark results.",
                "command": BenchstatCommand(self._logger, _command_tools, command_context).run
            },
			2: {
				"keys": ["pprof", "pp"],
                "description": "Starts the pprof tool.",
                "command": PprofInteractableCommand(self._logger, _command_tools, command_context).run
            },
            3: {
	            "keys": ["pprof-web", "pp-web"],
	             "description": "Starts the pprof tool with web interface.",
                "command": PprofWebCommand(self._logger, _command_tools, command_context).run
            }
		}

	def start(self) -> None:
		for _, command_info in self._commands_to_add.items():
			self._cli_interface.add_command(command_info["keys"], command_info["description"], command_info["command"])

		self._cli_interface.start_listen_loop()
