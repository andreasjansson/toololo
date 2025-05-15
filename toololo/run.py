# toololo/run.py
import time
import json
import traceback
from typing import Callable, Any, cast, Iterator
import anthropic

from .types import Output, ThinkingContent, TextContent, ToolUseContent, ToolResult
from .function import function_to_jsonschema, hashed_function_name, make_compatible


class Run:
    def __init__(
        self,
        client: anthropic.Client,
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
        self.tool_schemas = [
            function_to_jsonschema(client, model, func)
            for func in self.compatible_tools
        ]

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

    def __next__(self) -> Output:
        return next(self)

    def __iter__(self) -> Iterator[Output]:
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
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens + self.thinking_budget,
                        messages=self.messages,
                        tools=self.tool_schemas,
                        system=self.system,
                        thinking=self.thinking_dict,
                    )
                    break
                except anthropic.APIStatusError:
                    claude_attempt += 1
                    time.sleep(30)
                    if claude_attempt >= max_claude_attempts:
                        return

            # Process the response
            assistant_message_content = []
            has_tool_uses = False
            tool_results = []

            # Process each content item
            for content in response.content:
                assistant_message_content.append(content)

                if content.type == "thinking":
                    yield ThinkingContent(content.thinking)
                elif content.type == "text":
                    yield TextContent(content.text)
                elif content.type == "tool_use":
                    has_tool_uses = True
                    func_name = content.name
                    func_args = cast(dict[str, Any], content.input)

                    # Yield the tool use
                    yield ToolUseContent(content.name, func_args)

                    # Process the tool use
                    if func_name in self.function_map:
                        func = self.function_map[func_name]
                        original_func = self.original_function_map[func_name]
                        try:
                            result_content = json.dumps(func(**func_args))
                            success = True
                        except Exception as e:
                            result_content = "".join(traceback.format_exception(e))
                            success = False
                    else:
                        result_content = f"Invalid tool: {func_name}. Valid available tools are: {', '.join(self.function_map.keys())}"
                        original_func = None
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
            if not has_tool_uses:
                self.messages.append(
                    {"role": "assistant", "content": assistant_message_content}
                )
                return

            # Add the messages for the next iteration
            self.messages.append(
                {"role": "assistant", "content": assistant_message_content}
            )
            self.messages.append({"role": "user", "content": tool_results})

    def append_user_message(self, content):
        """
        Append a user message to be inserted at the next appropriate point in the conversation.
        The message will be added before the next API call to Claude.
        """
        self.pending_user_messages.append(content)
