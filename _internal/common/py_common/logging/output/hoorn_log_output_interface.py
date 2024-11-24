from abc import ABC, abstractmethod

from ...exceptions.invalid_operation_exception import InvalidOperationException
from ...logging.hoorn_log import HoornLog


class HoornLogOutputInterface(ABC):
    def __init__(self, is_child: bool = False):
        if is_child:
            return

        raise InvalidOperationException("An interface is not meant to be instantiated directly.")

    @abstractmethod
    def output(self, hoorn_log: HoornLog, encoding="utf-8") -> None:
        """
		Outputs a given hoorn log according to the specified implementation logic.
		:param encoding: An optional encoding for the output.
		:param hoorn_log: The hoorn log to output.
		:return: None.
		"""
        raise InvalidOperationException("You cannot call a method of an interface. Use a concrete implementation.")
