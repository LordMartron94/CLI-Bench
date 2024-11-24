from typing import List

import pydantic

from .argument_model import ArgumentModel


class MessagePayload(pydantic.BaseModel):
	action: str
	args: List[ArgumentModel]