from .python_profile_command import PythonProfileCommand
from ..common.py_common.cli_framework import CommandLineInterface
from ..common.py_common.logging import HoornLogger


class PythonCLI:
    def __init__(self, logger: HoornLogger, python_profile_command: PythonProfileCommand):
        self._cli_interface = CommandLineInterface(logger)
        self._logger: HoornLogger = logger

        self._commands_to_add = {
            0: {
                "keys": ["profile", "p"],
                "description": "Starts the profiling suite.",
                "command": python_profile_command.run
            }
        }

    def start(self) -> None:
        for _, command_info in self._commands_to_add.items():
            self._cli_interface.add_command(command_info["keys"], command_info["description"], command_info["command"])

        self._cli_interface.start_listen_loop()
