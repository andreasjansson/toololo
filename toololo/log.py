import logging
from typing import Protocol

from .types import Output

class MessageLogger(Protocol):
    def log_message(self, message: Output): ...


class DefaultMessageLogger(MessageLogger):
    
    def __init__(self):
        self.logger = logging.getLogger()
    
    def log_message(self, message: Output):
        self.logger.info(str(message))
