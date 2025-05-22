import asyncio
from typing import Protocol, Any, runtime_checkable

try:
    import anthropic

    has_anthropic = True
except ImportError:
    has_anthropic = False

try:
    import openai

    has_openai = True
except ImportError:
    has_openai = False


@runtime_checkable
class Client(Protocol):
    async def call(
        self,
        messages: list,
        tool_schemas: list[dict] | None = None,
        system_prompt: str | None = None,
    ) -> list: ...


class AnthropicClient(Client):
    def __init__(
        self,
        client: anthropic.AsyncClient,
        model: str,
        system_prompt: str | None = None,
        max_tokens=8192,
        thinking_budget: int = 4096,
        **kwargs,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.kwargs = kwargs

        if thinking_budget > 0:
            self.thinking_dict = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        else:
            self.thinking_dict = {"type": "disabled"}

        if system_prompt is None:
            system_prompt = ""
        else:
            system_prompt += "\n\n# Additional instructions\n"
        system_prompt += "Highly desirable: Whevener possible, call multiple tools in the same content block so that I can call the tools in parallel more efficiently."

        self.default_system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    async def call(
        self,
        messages: list,
        tool_schemas: list[dict] | None = None,
        system_prompt: str | None = None,
    ) -> list:
        if system_prompt is None:
            system = self.default_system
        else:
            system = system_prompt

        if tool_schemas is None:
            tool_schemas = []

        max_claude_attempts = 10
        claude_attempt = 0
        while claude_attempt < max_claude_attempts:
            try:
                return await self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens + self.thinking_budget,
                    messages=messages,
                    tools=tool_schemas,
                    system=system,
                    thinking=self.thinking_dict,
                    betas=["token-efficient-tools-2025-02-19"],
                    **self.kwargs,
                )
            except anthropic.APIStatusError as e:
                print(e)
                claude_attempt += 1
                await asyncio.sleep(30)
                if claude_attempt >= max_claude_attempts:
                    raise


class OpenAIClient(Client):
    def __init__(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        system_prompt: str | None = None,
        **kwargs,
    ):
        self.client = client
        self.model = model
        self.default_system_prompt = system_prompt
        self.kwargs = kwargs

    def call(
        self,
        messages: list,
        tool_schemas: list[dict] | None = None,
        system_prompt: str | None = None,
    ) -> list:
        if system_prompt is not None:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ] + messages

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tool_schemas,
            parallel_tool_calls=True,
            **self.kwargs,
        )


def make_client(client: Any, system_prompt: str | None = None, **client_kwargs) -> Client:
    if isinstance(client, Client):
        return client

    if has_anthropic and isinstance(client, anthropic.AsyncClient):
        return AnthropicClient(client, system_prompt=system_prompt, **client_kwargs)

    if has_openai and isinstance(client, openai.AsyncOpenAI):
        return OpenAIClient(client, system_prompt=system_prompt, **client_kwargs)

    raise ValueError(f"Unknown client class: {type(client)}. Must be one of anthropic.AsyncClient, openai.AsyncOpenAI")
