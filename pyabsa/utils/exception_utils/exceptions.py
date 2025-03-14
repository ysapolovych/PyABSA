"""Custom exceptions for PyABSA."""
class CheckpointLoadException(Exception):
    """Exception to warn about checpoints loading failure and the underying reason."""
    def __init__(self, checkpoint_path: str | None, message: str):
        self.checkpoint_path = checkpoint_path
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"Failed to load checkpoint from {self.checkpoint_path}. Error:\n{self.message}"