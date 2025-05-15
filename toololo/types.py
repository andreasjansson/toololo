from typing import Callable, Any, Protocol
from dataclasses import dataclass


class Output(Protocol):
    pass


@dataclass(frozen=True)
class ThinkingContent(Output):
    content: str

    def __repr__(self) -> str:
        return f"<<< THINKING >>>\n{self.content}"


@dataclass(frozen=True)
class TextContent(Output):
    content: str

    def __repr__(self) -> str:
        return f"<<< TEXT >>>\n{self.content}"


@dataclass(frozen=True)
class ToolUseContent(Output):
    name: str
    input: dict[str, Any]

    def __repr__(self) -> str:
        return f"<<< TOOL USE >>>\nFunction: {self.name}\nArguments: {self.input}"


@dataclass(frozen=True)
class ToolResult(Output):
    success: bool
    func: Callable[..., Any] | None
    content: Any

    def __repr__(self) -> str:
        return f"<<< TOOL RESULT >>>\n{self.content}"
