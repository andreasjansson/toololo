from typing import Callable, Any
from dataclasses import dataclass


@dataclass
class ThinkingContent:
    content: str

    def __str__(self) -> str:
        return f"<<< THINKING >>>\n{self.content}"


@dataclass
class TextContent:
    content: str

    def __str__(self) -> str:
        return f"<<< TEXT >>>\n{self.content}"


@dataclass
class ToolUseContent:
    name: str
    input: dict[str, Any]

    def __str__(self) -> str:
        return f"<<< TOOL USE >>>\nFunction: {self.name}\nArguments: {self.input}"


@dataclass
class ToolResult:
    success: bool
    func: Callable[..., Any] | None
    content: Any

    def __str__(self) -> str:
        return f"<<< TOOL RESULT >>>\n{self.content}"
