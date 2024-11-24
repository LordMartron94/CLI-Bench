from typing import Optional

import pydantic

from components.shared.MD_PyCommon.md_py_common.py_common.networking.message_payload import MessagePayload


class MessageModel(pydantic.BaseModel):
	payload: MessagePayload
	target_uuid: Optional[str]
