import pydantic


class ArgumentModel(pydantic.BaseModel):
	type: str
	value: str