from ..common.py_common.command_handling import CommandHelper
from ..common.py_common.handlers import FileHandler
from ..common.py_common.logging import HoornLogger
from ..common.py_common.user_input.user_input_helper import UserInputHelper


class CommandTools:
	"""
	"context" class to provide clients with low-level utility classes.
	"""

	def __init__(self, logger: HoornLogger):
		self._file_handler = FileHandler()
		self._command_handler = CommandHelper(logger)
		self._user_input_helper = UserInputHelper(logger)

	def get_file_handler(self) -> FileHandler:
		return self._file_handler

	def get_command_handler(self) -> CommandHelper:
		return self._command_handler

	def get_user_input_helper(self) -> UserInputHelper:
		return self._user_input_helper
