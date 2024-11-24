from ...logging.formatting.log_formatter_interface import HoornLogFormatterInterface
from ...logging.hoorn_log import HoornLog
from ...logging.log_type import LogType
from ...utils.color_helper import ColorHelper


class HoornLogColorFormatter(HoornLogFormatterInterface):
    """
    A formatter that formats log messages in a colorful way based on the log type.
    """
    def __init__(self):
        self._color_dict = {
            LogType.DEBUG: {
                "Text": "#079B00",
                "Background": None
            },
            LogType.INFO: {
                "Text": "#9B9B9B",
                "Background": None
            },
            LogType.WARNING: {
                "Text": "#FFA300",
                "Background": None
            },
            LogType.ERROR: {
                "Text": "#FF0000",
                "Background": None
            },
            LogType.CRITICAL: {
                "Text": "#FFFFFF",
                "Background": "#FF0000"
            },
            LogType.DEFAULT: {
                "Text": "#9B9B9B",
                "Background": None
            }
        }

        self._color_helper = ColorHelper()

        super().__init__(is_child=True)

    def format(self, hoorn_log: HoornLog) -> str:
        log_type = hoorn_log.log_type

        if hoorn_log.log_type not in self._color_dict:
            log_type = LogType.DEFAULT

        text_color_hex = self._color_dict[log_type]["Text"]
        background_color_hex = self._color_dict[log_type]["Background"]

        return self._color_helper.colorize_string(hoorn_log.formatted_message, text_color_hex, background_color_hex)
