from enum import Enum


class LogType(Enum):
    DEBUG = 0
    INFO = 1
    DEFAULT = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

    def __ge__(self, other):
        return self.value >= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.name)
