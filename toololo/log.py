import logging
from abc import ABC, abstractmethod

from .types import Output


class MessageLogger(ABC):
    def __init__(self):
        self.prefix = ""

    @abstractmethod
    def log_message(self, message: Output) -> None: ...

    @abstractmethod
    def clone(self) -> "MessageLogger": ...

    def with_appended_prefix(self, string: str, strip=True) -> "MessageLogger":
        clone = self.clone()
        clone.prefix += string
        if strip:
            clone.prefix = clone.prefix.strip()
        return clone

    def prefixed_message(self, message: Output) -> str:
        s = str(message)
        if self.prefix:
            s = f"{self.prefix}: {s}"
        return s


class DefaultMessageLogger(MessageLogger):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def log_message(self, message: Output) -> None:
        self.logger.info(self.prefixed_message(message))

    def clone(self) -> MessageLogger:
        logger = DefaultMessageLogger()
        logger.prefix = self.prefix
        return logger


class PrintMessageLogger(MessageLogger):
    def __init__(self):
        super().__init__()

    def log_message(self, message: Output) -> None:
        print(self.prefixed_message(message))

    def clone(self) -> MessageLogger:
        logger = PrintMessageLogger()
        logger.prefix = self.prefix
        return logger


default_message_logger = DefaultMessageLogger()
