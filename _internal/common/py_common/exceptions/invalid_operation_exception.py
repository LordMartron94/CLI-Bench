class InvalidOperationException(BaseException):
    """An exception that is thrown when an invalid operation is attempted."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
