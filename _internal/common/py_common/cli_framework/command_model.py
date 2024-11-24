from typing import List, Callable, Optional, Any

import pydantic


class CommandModel(pydantic.BaseModel):
	commands: List[str]
	description: str
	action: Callable
	arguments: Optional[List[Any]]
