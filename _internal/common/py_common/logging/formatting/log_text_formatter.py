from ...logging.formatting.log_formatter_interface import HoornLogFormatterInterface
from ...logging.hoorn_log import HoornLog
from ...logging.log_type import LogType


class HoornLogTextFormatter(HoornLogFormatterInterface):
    """
    Formats the text in the logs based on the maximum length of the log types.
    """

    def __init__(self):
        self._max_length = self._get_longest_log_type_length()

        super().__init__(is_child=True)

    def _get_longest_log_type_length(self) -> int:
        return max(len(log_type.name) for log_type in LogType)

    def format(self, hoorn_log: HoornLog) -> str:
        message = f"[{hoorn_log.log_time}] {hoorn_log.log_type.name} : {hoorn_log.log_message}"
        message = message.replace(hoorn_log.log_type.name, hoorn_log.log_type.name.ljust(self._max_length))

        return message
