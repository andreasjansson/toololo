from pathlib import Path
import json
import traceback
import asyncio
from typing import Callable, Any, cast, AsyncIterator

from .types import (
    Tool,
    Output,
    ThinkingContent,
    TextContent,
    ToolUseContent,
    ToolResult,
)
from .function import function_to_jsonschema, hashed_function_name, make_compatible
from .client import Client, make_client


class Run:
    def __init__(
        self,
        client: Any,
        messages: list | str,
        system_prompt: str | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        max_iterations=50,
        **client_kwargs,
    ):
        self.client = make_client(client, system_prompt, **client_kwargs)
        self.max_iterations = max_iterations
        self.functions_or_tools = tools

        if isinstance(messages, str):
            self.messages = [{"role": "user", "content": messages}]
        else:
            self.messages = encode_files(messages)

        self.pending_user_messages = []
        self.iteration = 0

        self.tool_map: dict[str, Tool] = {}
        self.initialized = False
        self._generator: AsyncIterator[Output] | None = None

    async def initialize(self) -> None:
        if self.initialized:
            return

        if self.functions_or_tools is not None:
            self.tools = await make_tool_map(self.client, self.functions_or_tools)
        self.initialized = True

    def __aiter__(self) -> AsyncIterator[Output]:
        """Return self as an async iterator."""
        return self

    async def __anext__(self) -> Output:
        return await self.get_generator().__anext__()

    def get_generator(self) -> AsyncIterator[Output]:
        """Get or create the async generator for iteration."""
        if self._generator is None:
            self._generator = self.generate_outputs()
        return self._generator

    async def generate_outputs(self) -> AsyncIterator[Output]:
        """Generate outputs as an async iterator."""

        # We can't run async functions in __init__ so we do async
        # initialization here lazily
        await self.initialize()

        for self.iteration in range(self.max_iterations):
            # Process any pending user messages that were added
            # by run.append_user_message
            if self.pending_user_messages:
                for message in self.pending_user_messages:
                    self.messages.append({"role": "user", "content": message})
                self.pending_user_messages = []

            tool_schemas = [tool.schema for tool in self.tool_map.values()]
            response = await self.client.call(
                messages=self.messages, tool_schemas=tool_schemas
            )
            assistant_message_content = []
            tool_use_tasks = []
            tool_results = []

            for content in response.content:
                assistant_message_content.append(content)

                if content.type == "thinking":
                    yield ThinkingContent(content.thinking)
                elif content.type == "text":
                    yield TextContent(content.text)
                elif content.type == "tool_use":
                    func_name = content.name
                    func_args = cast(dict[str, Any], content.input)

                    yield ToolUseContent(content.name, func_args)

                    if func_name in self.tool_map:
                        tool = self.tool_map[func_name]
                        task = tool.call(**func_args)
                        tool_use_tasks.append((content, task, tool.func, True))
                    else:
                        error_msg = f"Invalid tool: {func_name}. Valid available tools are: {', '.join(self.tool_map.keys())}"
                        tool_use_tasks.append((content, error_msg, None, False))

            # Execute all tool calls in parallel if there are any
            if tool_use_tasks:
                tool_results = []
                for content, task_or_error, func, is_task in tool_use_tasks:
                    if is_task:
                        try:
                            result = await task_or_error
                            result_with_encoded_files = encode_files(result)
                            result_content = json.dumps(result_with_encoded_files)
                            success = True
                        except Exception as e:
                            result_content = "".join(traceback.format_exception(e))
                            success = False
                    else:
                        result_content = task_or_error
                        success = False

                    yield ToolResult(success, func, result_content)

                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": result_content,
                    }

                    if len(result_content) >= 1_000:
                        clear_previous_cache_control(self.messages)
                        tool_result["cache_control"] = {"type": "ephemeral"}

                    tool_results.append(tool_result)

            # If no tool uses, we're done
            else:
                self.messages.append(
                    {"role": "assistant", "content": assistant_message_content}
                )
                return

            self.messages += [
                {"role": "assistant", "content": assistant_message_content},
                {"role": "user", "content": tool_results},
            ]

    def append_user_message(self, content):
        """
        Append a user message to be inserted at the next appropriate point in the conversation.
        The message will be added before the next API call to Claude.
        """
        self.pending_user_messages.append(content)


async def make_tool_map(
    client: Client,
    functions_or_tools: list[Callable[..., Any] | Tool],
) -> dict[str, Tool]:
    # Make all tool schemas in parallel
    tasks = []
    for func_or_tool in functions_or_tools:
        tasks.append(make_tool(client, func_or_tool))

    tool_map = {}
    for tool in await asyncio.gather(*tasks):
        tool_map[tool.name] = tool

    return tool_map


async def make_tool(client: Client, func_or_tool: Callable[..., Any] | Tool) -> Tool:
    if isinstance(func_or_tool, Tool):
        tool = func_or_tool
    else:
        func = func_or_tool
        schema = await function_to_jsonschema(client, func)
        tool = Tool(
            name=hashed_function_name(func),
            func=func_or_tool,
            wrapped_func=make_compatible(func),
            schema=schema,
        )
    assert len(tool.name) <= 64
    return tool


def clear_previous_cache_control(messages) -> None:
    for message in messages:
        message_content = message.get("content", [])
        if isinstance(message_content, list):
            for tr in message_content:
                if isinstance(tr, dict) and "cache_control" in tr:
                    del tr["cache_control"]


def encode_files(obj: Any) -> Any:
    return walk_object(obj, maybe_encode_file)


def maybe_encode_file(x: Any) -> Any:
    if isinstance(x, Path):
        return encode_file(x)
    return x


def encode_file(x: Path) -> dict:
    pass


def walk_object(obj: Any, fn: Callable[..., Any]) -> Any:
    if isinstance(obj, dict):
        return {k: walk_object(v, fn) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [walk_object(item, fn) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(walk_object(item, fn) for item in obj)
    else:
        return fn(obj)
