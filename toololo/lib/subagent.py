"""Parallel subagent execution for toololo standard library."""

import logging
from typing import Any, Callable, AsyncIterator
import aiostream
import openai

from ..run import Run
from ..types import Output, TextContent
from ..log import default_message_logger, MessageLogger

logger = logging.getLogger(__name__)


async def iterate_outputs(agents: list[Run]) -> AsyncIterator[tuple[int, Output]]:
    """Iterate over outputs from multiple agents running in parallel."""
    if not agents:
        return

    async def tagged_stream(agent_idx: int, agent: Run):
        try:
            async for output in agent:
                yield agent_idx, output
                logger.debug(
                    f"Agent {agent_idx} produced output: {type(output).__name__}"
                )
        except Exception as e:
            logger.error(f"Agent {agent_idx} failed: {e}")
        finally:
            logger.info(f"Agent {agent_idx} completed")

    # Merge all agent streams
    streams = [tagged_stream(i, agent) for i, agent in enumerate(agents)]
    async with aiostream.stream.merge(*streams).stream() as merged:
        async for agent_idx, output in merged:
            yield agent_idx, output

    logger.info("All agents completed")


class ParallelSubagents:
    """Manager for running multiple subagents in parallel."""

    def __init__(
        self,
        client: openai.AsyncOpenAI,
        tools: list[Callable[..., Any]] | None = None,
        model: str = "openai/gpt-5-mini",
        system_prompt: str | None = None,
        max_tokens: int = 8192,
        reasoning_max_tokens: int | None = None,
        max_iterations: int = 50,
    ):
        self.client = client
        self.tools = tools or []
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.reasoning_max_tokens = reasoning_max_tokens
        self.max_iterations = max_iterations

    async def spawn_agents(
        self,
        agent_prompts: list[str],
        _toololo_message_logger: MessageLogger = default_message_logger,
    ) -> list[str]:
        """Spawn multiple subagents and return their final assistant messages.

        Args:
            agent_prompts: List of prompts

        Returns:
            List of final assistant message contents from each completed agent
        """
        # Runtime validation to catch common mistake
        if not isinstance(agent_prompts, list):
            raise TypeError(
                f"agent_prompts must be a list of strings, got {type(agent_prompts)}"
            )

        for i, prompt in enumerate(agent_prompts):
            if not isinstance(prompt, str):
                raise TypeError(
                    f"agent_prompts[{i}] must be a string, got {type(prompt)}: {repr(prompt)}"
                )

        logger.info(f"Spawning {len(agent_prompts)} parallel subagents")

        # Create agents
        agents = []
        for i, prompt in enumerate(agent_prompts):
            # Create the agent with stored tools
            agent = Run(
                client=self.client,
                messages=prompt,
                model=self.model,
                tools=self.tools,
                system_prompt=self.system_prompt,
                max_tokens=self.max_tokens,
                reasoning_max_tokens=self.reasoning_max_tokens,
                max_iterations=self.max_iterations,
                message_logger=_toololo_message_logger.with_appended_prefix(f" subagent {i}"),
            )
            agents.append(agent)
            logger.info(f"Created subagent {i} with {len(self.tools)} tools")

        # Run all agents in parallel and collect final messages
        final_messages = [""] * len(agents)
        async for agent_index, output in iterate_outputs(agents):
            # Collect final assistant messages (TextContent outputs)
            if isinstance(output, TextContent):
                logger.debug(
                    f"Agent {agent_index} TextContent type: {type(output.content)}, value: {repr(output.content[:100])}"
                )
                final_messages[agent_index] = output.content

        logger.info(
            f"Collected {len(final_messages)} final messages from spawned agents"
        )
        logger.debug(
            f"Final messages types: {[type(msg) for msg in final_messages[:5]]}"
        )
        logger.debug(
            f"Final messages preview: {[repr(msg) for msg in final_messages[:5]]}"
        )
        return final_messages
