import time
import json
import traceback
from typing import Callable, Any, cast
import anthropic

from .types import ThinkingContent, TextContent, ToolUseContent, ToolResult
from .function import function_to_jsonschema, hashed_function_name, make_compatible


def run(
    client: anthropic.Client,
    messages: list | str,
    model: str,
    tools: list[Callable[..., Any]],
    system_prompt: str = "",
    max_tokens=8192,
    thinking_budget: int = 4096,
    max_iterations=50,
):
    if thinking_budget > 0:
        thinking_dict = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
    else:
        thinking_dict = {"type": "disabled"}

    compatible_tools = [make_compatible(func) for func in tools]
    function_map = {hashed_function_name(func): func for func in compatible_tools}
    original_function_map = {
        hashed_function_name(compatible_func): func
        for func, compatible_func in zip(tools, compatible_tools)
    }
    tool_schemas = [
        function_to_jsonschema(client, model, func) for func in compatible_tools
    ]

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if system_prompt:
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system = []

    for iteration in range(max_iterations):
        max_claude_attempts = 10
        claude_attempt = 0
        while claude_attempt < max_claude_attempts:
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens + thinking_budget,
                    messages=messages,
                    tools=tool_schemas,
                    system=system,
                    thinking=thinking_dict,
                )
                break
            except anthropic.APIStatusError as e:
                time.sleep(30)


        assistant_message_content = []
        has_tool_uses = False
        tool_results = []

        for content in response.content:
            assistant_message_content.append(content)
            if content.type == "thinking":
                yield ThinkingContent(content.thinking)
            if content.type == "text":
                yield TextContent(content.text)
            elif content.type == "tool_use":
                has_tool_uses = True
                func_name = content.name
                func_args = cast(dict[str, Any], content.input)
                yield ToolUseContent(content.name, func_args)

                if func_name in function_map:
                    func = function_map[func_name]
                    original_func = original_function_map[func_name]
                    try:
                        result_content = json.dumps(func(**func_args))
                        success = True
                    except Exception as e:
                        result_content = "".join(traceback.format_exception(e))
                        success = False
                else:
                    result_content = f"Invalid tool: {func_name}. Valid available tools are: {', '.join(function_map.keys())}"
                    original_func = None

                yield ToolResult(success, original_func, result_content)

                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": content.id,
                    "content": result_content,
                }

                if len(result_content) >= 1_000:
                    for message in messages:
                        content = message["content"]
                        if isinstance(content, list):
                            for tr in content:
                                if "cache_control" in tr:
                                    del tr["cache_control"]
                    tool_result["cache_control"] = {"type": "ephemeral"}

                tool_results.append(tool_result)

        # If no tool uses, we're done
        if not has_tool_uses:
            return

        new_messages = [
            {"role": "assistant", "content": assistant_message_content},
            {"role": "user", "content": tool_results},
        ]

        messages += new_messages

    raise Exception(
        f"Failed to generate a successful response after {max_iterations} iterations"
    )
