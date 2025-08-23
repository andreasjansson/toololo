"""Parallel subagent execution for toololo standard library."""

import asyncio
import logging
from typing import Any, Callable, AsyncIterator, Optional
from dataclasses import dataclass
import openai

from ..run import Run
from ..types import Output, TextContent

logger = logging.getLogger(__name__)


async def iterate_outputs(agents: list[Run]) -> AsyncIterator[tuple[int, Output]]:
    """Iterate over outputs from multiple agents running in parallel."""
    if not agents:
        return
    
    # Create async iterators for each agent
    agent_iterators = {i: agent.__aiter__() for i, agent in enumerate(agents)}
    active_agents = set(range(len(agents)))
    running_tasks = {}
    
    logger.info(f"Starting parallel execution of {len(agents)} agents")
    
    # Start initial tasks
    for agent_idx in active_agents:
        if agent_idx not in running_tasks:
            agent_iter = agent_iterators[agent_idx]
            task = asyncio.create_task(agent_iter.__anext__())
            running_tasks[agent_idx] = task
    
    while active_agents:
        if not running_tasks:
            break
        
        # Wait for at least one task to complete
        done, pending = await asyncio.wait(
            running_tasks.values(),
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Process completed tasks
        for task in done:
            # Find which agent this task belongs to
            agent_idx = None
            for idx, t in running_tasks.items():
                if t == task:
                    agent_idx = idx
                    break
            
            if agent_idx is None:
                continue
                
            # Remove completed task
            del running_tasks[agent_idx]
            
            try:
                output = await task
                yield agent_idx, output
                logger.debug(f"Agent {agent_idx} produced output: {type(output).__name__}")
                
                # Start next task for this agent if still active
                if agent_idx in active_agents:
                    agent_iter = agent_iterators[agent_idx]
                    next_task = asyncio.create_task(agent_iter.__anext__())
                    running_tasks[agent_idx] = next_task
                
            except StopAsyncIteration:
                # Agent finished
                active_agents.remove(agent_idx)
                logger.info(f"Agent {agent_idx} completed")
                
            except Exception as e:
                # Agent errored
                active_agents.remove(agent_idx)
                logger.error(f"Agent {agent_idx} failed: {e}")
    
    # Cancel any remaining tasks
    for task in running_tasks.values():
        task.cancel()
    
    logger.info("All agents completed")


@dataclass
class SubagentOutput:
    """Output from a subagent with metadata."""
    agent_index: int
    agent_id: str
    output: Output
    is_final: bool = False
    error: Optional[str] = None


class ParallelSubagents:
    """Manager for running multiple subagents in parallel."""
    
    def __init__(
        self,
        client: openai.AsyncOpenAI,
        tools: list[Callable[..., Any]] | None = None,
        model: str = "openai/gpt-5-mini",
        max_tokens: int = 8192,
        reasoning_max_tokens: Optional[int] = None,
        max_iterations: int = 50
    ):
        self.client = client
        self.tools = tools or []
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_max_tokens = reasoning_max_tokens
        self.max_iterations = max_iterations
    
    async def spawn_agents(
        self,
        agent_prompts: list[str],
        system_prompt: str = ""
    ) -> list[str]:
        """Spawn multiple subagents and return their final assistant messages.
        
        Args:
            agent_prompts: List of prompts, or list of (system_prompt, prompt) tuples
            system_prompt: Default system prompt(s) to use. If string, used for all agents.
                         If list, must match length of agent_prompts (ignored for tuples).
            
        Returns:
            List of final assistant message contents from each completed agent
        """
        logger.info(f"Spawning {len(agent_prompts)} parallel subagents")
        
        # Create agents
        agents = []
        for i, item in enumerate(agent_prompts):
            # Handle different input formats
            if isinstance(item, tuple):
                # (system_prompt, prompt) tuple
                sys_prompt, prompt = item
            else:
                # Just a prompt string
                prompt = item
                if isinstance(system_prompt, list):
                    sys_prompt = system_prompt[i] if i < len(system_prompt) else ""
                else:
                    sys_prompt = system_prompt
            
            # Create the agent with stored tools
            agent = Run(
                client=self.client,
                messages=prompt,
                model=self.model,
                tools=self.tools,
                system_prompt=sys_prompt,
                max_tokens=self.max_tokens,
                reasoning_max_tokens=self.reasoning_max_tokens,
                max_iterations=self.max_iterations
            )
            agents.append(agent)
            logger.info(f"Created subagent {i} with {len(self.tools)} tools")
        
        # Run all agents in parallel and collect final messages
        final_messages = []
        async for agent_index, output in iterate_outputs(agents):
            # Collect final assistant messages (TextContent outputs)
            if isinstance(output, TextContent):
                final_messages.append(output.content)
                logger.debug(f"Agent {agent_index} produced text: {output.content[:100]}...")
        
        logger.info(f"Collected {len(final_messages)} final messages from spawned agents")
        return final_messages
    




