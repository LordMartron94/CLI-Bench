from abc import ABC, abstractmethod

from ...exceptions.invalid_operation_exception import InvalidOperationException
from ...logging.hoorn_log import HoornLog


class HoornLogFormatterInterface(ABC):
    """
    A base class for formatting log messages.
    """

    def __init__(self, is_child: bool = False):
        if is_child:
            return

        raise InvalidOperationException("You cannot instantiate an interface. Use a concrete implementation.")

    @abstractmethod
    def format(self, log: HoornLog) -> str:
        """Formats the log message based on the behavior of the concrete implementation."""
        raise InvalidOperationException("You are attempting to call the method of an interface directly, use the "
                                        "concrete implementation.")