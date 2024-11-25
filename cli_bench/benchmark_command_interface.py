from abc import abstractmethod

from .common.py_common.logging import HoornLogger


class IBenchmarkCommand:
	"""
	Interface for benchmark commands.
	"""

	def __init__(self, logger: HoornLogger, is_child: bool = False):
		if not is_child:
			raise ValueError("You cannot instantiate an interface. Use a concrete implementation.")

		self._logger: HoornLogger = logger

	@abstractmethod
	def run(self) -> None:
		"""Runs the command."""
		raise ValueError("Subclasses must implement this method. Don't call it directly. (fool).")

