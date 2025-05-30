import json
from typing import Callable, Any
import asyncio
import anthropic

from .run import Run
from .types import ToolResult


def truncate_lines(text: str, max_lines: int = 50, max_line_length: int = 200) -> str:
    """Truncate text to a maximum number of lines and line length for display."""
    lines = text.split("\n")

    # Truncate each line if it's too long
    truncated_lines = []
    for line in lines[:max_lines]:
        if len(line) > max_line_length:
            truncated_lines.append(line[:max_line_length] + "...")
        else:
            truncated_lines.append(line)

    # Add indication if we truncated lines
    if len(lines) > max_lines:
        truncated_lines.append(f"... ({len(lines) - max_lines} more lines)")

    return "\n".join(truncated_lines)


def print_help():
    """Print help information for the chat interface."""
    help_text = """
Available commands:
  /help      - Show this help message
  /debug     - Show conversation history as JSON
  q, quit, exit - Exit the chat

Usage:
  - Type your message and press Enter to chat with the AI
  - The AI has access to the provided tools and can use them to help you
  - Use Ctrl+C to interrupt generation
"""
    print(help_text.strip())


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            obj_dict = obj.to_dict()
            if "signature" in obj_dict:
                del obj_dict["signature"]
            return obj_dict
        return super().default(obj)


async def chat(
    tools: list[Callable[..., Any]],
    model="claude-sonnet-4-20250514",
    max_iterations=200,
):
    anthropic_client = anthropic.AsyncAnthropic()

    messages = []

    while True:
        print("> ", end="")
        prompt = input()
        if not prompt:
            continue

        if prompt.strip().lower() in ["q", "quit", "exit"]:
            return

        if prompt.strip().lower() == "/help":
            print_help()
            continue

        if prompt.strip().lower() == "/debug":
            print(json.dumps(messages, indent=2, cls=CustomJSONEncoder))
            continue

        run = Run(
            client=anthropic_client,
            model=model,
            messages=messages + [{"role": "user", "content": prompt}],
            tools=tools,
            max_iterations=max_iterations,
        )
        try:
            async for output in run:
                if isinstance(output, ToolResult):
                    print(truncate_lines(str(output)))
                else:
                    print(output)
                    print()
        except (KeyboardInterrupt, asyncio.exceptions.CancelledError):
            print("\n[Generation interrupted]")

        messages = run.messages
