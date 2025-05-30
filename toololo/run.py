import json
import traceback
import asyncio
import inspect
from typing import Callable, Any, cast, AsyncIterator, Optional
import anthropic

from .types import Output, ThinkingContent, TextContent, ToolUseContent, ToolResult
from .function import function_to_jsonschema, hashed_function_name, make_compatible


class Run:
    def __init__(
        self,
        client: anthropic.AsyncClient,
        messages: list | str,
        model: str,
        tools: list[Callable[..., Any]],
        system_prompt: str = "",
        max_tokens=8192,
        thinking_budget: int = 4096,
        max_iterations=50,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.max_iterations = max_iterations

        if thinking_budget > 0:
            self.thinking_dict = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        else:
            self.thinking_dict = {"type": "disabled"}

        self.compatible_tools = [make_compatible(func) for func in tools]
        self.function_map = {
            hashed_function_name(func): func for func in self.compatible_tools
        }
        self.original_function_map = {
            hashed_function_name(compatible_func): func
            for func, compatible_func in zip(tools, self.compatible_tools)
        }
        self.tool_schemas = []

        if isinstance(messages, str):
            self.messages = [{"role": "user", "content": messages}]
        else:
            self.messages = messages.copy()

        if system_prompt:
            self.system = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            self.system = []

        self.pending_user_messages = []
        self.iteration = 0

        self.initialized = False
        self._generator: Optional[AsyncIterator[Output]] = None

    async def initialize(self) -> None:
        if self.initialized:
            return

        # Execute all function_to_jsonschema calls in parallel
        tasks = [
            function_to_jsonschema(self.client, self.model, func)
            for func in self.compatible_tools
        ]
        self.tool_schemas = await asyncio.gather(*tasks)
        self.initialized = True

    def __aiter__(self) -> AsyncIterator[Output]:
        """Return self as an async iterator."""
        return self

    async def __anext__(self) -> Output:
        return await self._get_generator().__anext__()

    def _get_generator(self) -> AsyncIterator[Output]:
        """Get or create the async generator for iteration."""
        if self._generator is None:
            self._generator = self._generate_outputs()
        return self._generator

    async def _generate_outputs(self) -> AsyncIterator[Output]:
        """Generate outputs as an async iterator."""
        await self.initialize()
        for self.iteration in range(self.max_iterations):
            # Process any pending user messages
            if self.pending_user_messages:
                for message in self.pending_user_messages:
                    self.messages.append({"role": "user", "content": message})
                self.pending_user_messages = []

            # Get response from Claude
            max_claude_attempts = 10
            claude_attempt = 0
            while claude_attempt < max_claude_attempts:
                try:
                    response = await self.client.beta.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens + self.thinking_budget,
                        messages=self.messages,
                        tools=self.tool_schemas,
                        system=self.system,
                        thinking=self.thinking_dict,
                        betas=["token-efficient-tools-2025-02-19"],
                    )
                    break
                except anthropic.APIStatusError:
                    claude_attempt += 1
                    await asyncio.sleep(30)
                    if claude_attempt >= max_claude_attempts:
                        return

            # Process the response
            assistant_message_content = []
            tool_results = []

            # Find all tool_use blocks for parallel processing
            tool_use_tasks = []
            tool_use_contents = []

            # First pass: collect all content items and prepare tool calls
            for content in response.content:
                assistant_message_content.append(content)

                if content.type == "thinking":
                    yield ThinkingContent(content.thinking)
                elif content.type == "text":
                    yield TextContent(content.text)
                elif content.type == "tool_use":
                    func_name = content.name
                    func_args = cast(dict[str, Any], content.input)

                    # Yield the tool use
                    tool_content = ToolUseContent(content.name, func_args)
                    yield tool_content
                    tool_use_contents.append((content, tool_content))

                    # Create task for parallel execution
                    if func_name in self.function_map:
                        func = self.function_map[func_name]
                        original_func = self.original_function_map[func_name]
                        task = self._execute_function(func, **func_args)
                        tool_use_tasks.append((content, task, original_func, True))
                    else:
                        error_msg = f"Invalid tool: {func_name}. Valid available tools are: {', '.join(self.function_map.keys())}"
                        tool_use_tasks.append((content, error_msg, None, False))

            # Execute all tool calls in parallel if there are any
            if tool_use_tasks:
                # Wait for all tasks to complete (or error)
                tool_results = []
                for content, task_or_error, original_func, is_task in tool_use_tasks:
                    if is_task:
                        try:
                            # Execute the task
                            result = await task_or_error
                            result_content = json.dumps(result)
                            success = True
                        except Exception as e:
                            result_content = "".join(traceback.format_exception(e))
                            success = False
                    else:
                        # This is already an error message
                        result_content = task_or_error
                        success = False

                    # Yield the tool result
                    yield ToolResult(success, original_func, result_content)

                    # Prepare the tool result for Claude
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": result_content,
                    }

                    if len(result_content) >= 1_000:
                        for message in self.messages:
                            message_content = message.get("content", [])
                            if isinstance(message_content, list):
                                for tr in message_content:
                                    if isinstance(tr, dict) and "cache_control" in tr:
                                        del tr["cache_control"]
                        tool_result["cache_control"] = {"type": "ephemeral"}

                    tool_results.append(tool_result)

            # If no tool uses, we're done
            else:
                self.messages.append(
                    {"role": "assistant", "content": assistant_message_content}
                )
                return

            # Add the messages for the next iteration
            self.messages += [
                {"role": "assistant", "content": assistant_message_content},
                {"role": "user", "content": tool_results},
            ]

    async def _execute_function(self, func, **kwargs):
        """Execute a function, handling both sync and async functions appropriately"""
        if inspect.iscoroutinefunction(func):
            # Async function - await it directly
            return await func(**kwargs)
        else:
            # Sync function - run in an executor to avoid blocking
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: func(**kwargs)
            )

    def append_user_message(self, content):
        """
        Append a user message to be inserted at the next appropriate point in the conversation.
        The message will be added before the next API call to Claude.
        """
        self.pending_user_messages.append(content)
