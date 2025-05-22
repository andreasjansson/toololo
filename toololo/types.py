import inspect
import asyncio
from typing import Callable, Any, Protocol
from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    func: Callable[..., Any]
    schema: dict[str, Any]
    wrapped_func: Callable[..., Any] | None = None

    async def call(self, **kwargs):
        if self.wrapped_func is None:
            func = self.func
        else:
            func = self.wrapped_func

        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: func(**kwargs)
            )


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
