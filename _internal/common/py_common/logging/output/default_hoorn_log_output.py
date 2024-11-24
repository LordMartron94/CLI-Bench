from ...logging.hoorn_log import HoornLog
from ...logging.output.hoorn_log_output_interface import HoornLogOutputInterface


class DefaultHoornLogOutput(HoornLogOutputInterface):
    def __init__(self, max_separator_length: int = 30):
        self._max_separator_length = max_separator_length
        super().__init__(is_child=True)

    def output(self, hoorn_log: HoornLog, encoding="utf-8") -> None:
        if "${ignore=default}" in hoorn_log.formatted_message:
            return

        print(f"[{hoorn_log.separator:<{self._max_separator_length}}] {hoorn_log.formatted_message}")
