from typing import List, Union

from colorama import init

from ..logging.factory.hoorn_log_factory import HoornLogFactory
from ..logging.log_type import LogType
from ..logging.output.default_hoorn_log_output import DefaultHoornLogOutput
from ..logging.output.hoorn_log_output_interface import HoornLogOutputInterface


class HoornLogger:
    def __init__(self, 
                 outputs: Union[List[HoornLogOutputInterface], None] = None,
                 min_level: LogType = LogType.INFO,
                 separator_root: str = "",
                 max_separator_length: int = 30):
        """
        Initializes a new instance of the HoornLogger class.
        :param outputs: The output methods to use.
        Defaults to :class:`DefaultHoornLogOutput` which is the console.
        :param min_level: The minimum log level to output.
        Defaults to :class:`LogType.INFO`.
        """

        init(autoreset=True)

        if outputs is None or len(outputs) == 0:
            outputs = [DefaultHoornLogOutput(max_separator_length=max_separator_length)]

        self._log_factory = HoornLogFactory(max_separator_length=max_separator_length)

        self._min_level = min_level
        self._outputs = outputs
        self._separator_root = separator_root

    def set_min_level(self, min_level: LogType) -> None:
        """
        Sets the minimum log level to output.
        :param min_level: The minimum log level to output.
        """
        self._min_level = min_level

    def _can_output(self, log_type: LogType) -> bool:
        return log_type >= self._min_level

    def _log(self, log_type: LogType, message: str, encoding: str, force_show: bool = False, separator: str = None) -> None:
        if not force_show and not self._can_output(log_type):
            return

        hoorn_log = self._log_factory.create_hoorn_log(log_type, message, separator=self._separator_root + f".{separator}" if separator else self._separator_root)

        for output in self._outputs:
            output.output(hoorn_log, encoding=encoding)

    def debug(self, message: str, force_show: bool = False, encoding="utf-8", separator: str = None) -> None:
        """
        Logs a debug message.
        :param message: The message that is to be logged.
        :param force_show: Optional: Bypass the minlog setting.
        :param encoding: Optional: The encoding to use for the message.
        :param separator: Optional: A separator for the log message...
        Can see it as a kind of identifier.
        Each output interface uses this differently.
        Check the specific implementation for more details.
        By default, it is appended to the separator root.
        :return: None
        """

        self._log(LogType.DEBUG, message, force_show=force_show, encoding=encoding, separator=separator)

    def info(self, message: str, force_show: bool = False, encoding="utf-8", separator: str = None) -> None:
        """
        Logs an info message.
        :param message: The message that is to be logged.
        :param force_show: Optional: Bypass the minlog setting.
        :param encoding: Optional: The encoding to use for the message.
        :param separator: Optional: A separator for the log message...
        Can see it as a kind of identifier.
        Each output interface uses this differently.
        Check the specific implementation for more details.
        By default, it is appended to the separator root.
        :return: None
        """

        self._log(LogType.INFO, message, force_show=force_show, encoding=encoding, separator=separator)

    def warning(self, message: str, force_show: bool = False, encoding="utf-8", separator: str = None) -> None:
        """
        Logs a warning message.
        :param message: The message that is to be logged.
        :param force_show: Optional: Bypass the minlog setting.
        :param encoding: Optional: The encoding to use for the message.
        :param separator: Optional: A separator for the log message...
        Can see it as a kind of identifier.
        Each output interface uses this differently.
        Check the specific implementation for more details.
        By default, it is appended to the separator root.
        :return: None
        """

        self._log(LogType.WARNING, message, force_show=force_show, encoding=encoding, separator=separator)

    def error(self, message: str, force_show: bool = False, encoding="utf-8", separator: str = None) -> None:
        """
        Logs an error message.
        :param message: The message that is to be logged.
        :param force_show: Optional: Bypass the minlog setting.
        :param encoding: Optional: The encoding to use for the message.
        :param separator: Optional: A separator for the log message...
        Can see it as a kind of identifier.
        Each output interface uses this differently.
        Check the specific implementation for more details.
        By default, it is appended to the separator root.
        :return: None
        """

        self._log(LogType.ERROR, message, force_show=force_show, encoding=encoding, separator=separator)

    def critical(self, message: str, force_show: bool = False, encoding="utf-8", separator: str = None) -> None:
        """
        Logs a critical message.
        :param message: The message that is to be logged.
        :param force_show: Optional: Bypass the minlog setting.
        :param encoding: Optional: The encoding to use for the message.
        :param separator: Optional: A separator for the log message...
        Can see it as a kind of identifier.
        Each output interface uses this differently.
        Check the specific implementation for more details.
        By default, it is appended to the separator root.
        :return: None
        """

        self._log(LogType.CRITICAL, message, force_show=force_show, encoding=encoding, separator=separator)
