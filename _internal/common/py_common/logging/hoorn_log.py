from datetime import datetime
from typing import Optional

import pydantic

from ..logging.log_type import LogType


class HoornLog(pydantic.BaseModel):
    log_time: datetime
    log_type: LogType
    log_message: str
    formatted_message: Optional[str] = None
    separator: Optional[str] = ""

    def __str__(self) -> str:
        return self.formatted_message
