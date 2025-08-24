import logging
from typing import Protocol

from .types import Output, ThinkingContent, TextContent, ToolUseContent, ToolResult

class MessageLogger(Protocol):
    def log_message(self, message: Output): ...


class DefaultMessageLogger(MessageLogger):
    
    def __init__(self, logger_name: str = "toololo", level: int = logging.INFO):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(level)
    
    def log_message(self, message: Output):
        if isinstance(message, ThinkingContent):
            self.logger.debug(f"THINKING: {message.content}")
        elif isinstance(message, TextContent):
            self.logger.info(f"TEXT: {message.content}")
        elif isinstance(message, ToolUseContent):
            self.logger.info(f"TOOL USE - {message.name}: {message.input}")
        elif isinstance(message, ToolResult):
            level = logging.INFO if message.success else logging.WARNING
            status = "SUCCESS" if message.success else "FAILURE"
            self.logger.log(level, f"TOOL RESULT - {status}: {message.content}")
        else:
            self.logger.info(f"OUTPUT: {message}")
