"""Parallel subagent execution for toololo standard library."""

import asyncio
import logging
from typing import Any, Callable, AsyncIterator, Optional
from functools import partial
from dataclasses import dataclass
import openai

from ..run import Run
from ..types import Output

logger = logging.getLogger(__name__)


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
        model: str = "gpt-4o-mini",
        tools: Optional[list[Callable[..., Any]]] = None,
        max_tokens: int = 8192,
        reasoning_max_tokens: Optional[int] = None,
        max_iterations: int = 50
    ):
        self.client = client
        self.model = model
        self.tools = tools or []
        self.max_tokens = max_tokens
        self.reasoning_max_tokens = reasoning_max_tokens
        self.max_iterations = max_iterations
        self._agents: list[Run] = []
        self._agent_ids: list[str] = []
    
    def _bind_client_to_tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Bind the OpenAI client to tools that need it using partial application."""
        bound_tools = []
        
        for tool in tools:
            import inspect
            sig = inspect.signature(tool)
            
            # Check if the tool expects a 'client' parameter
            if 'client' in sig.parameters:
                # Bind the client using partial
                bound_tool = partial(tool, client=self.client)
                # Copy over the function metadata
                bound_tool.__name__ = tool.__name__
                bound_tool.__doc__ = tool.__doc__
                if hasattr(tool, '__annotations__'):
                    bound_tool.__annotations__ = tool.__annotations__
                bound_tools.append(bound_tool)
            else:
                # Tool doesn't need client, use as-is
                bound_tools.append(tool)
        
        return bound_tools
    
    async def spawn_agents(
        self,
        agent_prompts: list[str],
        system_prompt: str | list[str] = ""
    ) -> AsyncIterator[SubagentOutput]:
        """Spawn multiple subagents and yield their outputs as they come.
        
        Args:
            agent_prompts: List of prompts for each agent
            system_prompt: System prompt for all agents, or list of system prompts (one per agent)
            
        Yields:
            SubagentOutput containing outputs from each subagent as they execute
        """
        logger.info(f"Spawning {len(agent_prompts)} parallel subagents")
        
        # Handle system prompts
        if isinstance(system_prompt, str):
            system_prompts = [system_prompt] * len(agent_prompts)
        else:
            system_prompts = system_prompt
            if len(system_prompts) != len(agent_prompts):
                raise ValueError(f"Number of system prompts ({len(system_prompts)}) must match number of agent prompts ({len(agent_prompts)})")
        
        # Create agents
        self._agents = []
        self._agent_ids = []
        
        # Bind client to tools that need it
        bound_tools = self._bind_client_to_tools(self.tools)
        
        for i, (prompt, sys_prompt) in enumerate(zip(agent_prompts, system_prompts)):
            agent_id = f"agent_{i}_{hash((sys_prompt, prompt))}"
            self._agent_ids.append(agent_id)
            
            # Create the agent
            agent = Run(
                client=self.client,
                messages=prompt,
                model=self.model,
                tools=bound_tools,
                system_prompt=sys_prompt,
                max_tokens=self.max_tokens,
                reasoning_max_tokens=self.reasoning_max_tokens,
                max_iterations=self.max_iterations
            )
            self._agents.append(agent)
            logger.info(f"Created subagent {i} ({agent_id}) with {len(bound_tools)} tools")
        
        # Run all agents in parallel and yield outputs as they come
        async for output in self._run_agents_parallel():
            yield output
    
    async def _run_agents_parallel(self) -> AsyncIterator[SubagentOutput]:
        """Run all agents in parallel and yield outputs."""
        if not self._agents:
            return
        
        # Create async iterators for each agent
        agent_iterators = [(i, agent.__aiter__()) for i, agent in enumerate(self._agents)]
        active_agents = set(range(len(self._agents)))
        
        logger.info(f"Starting parallel execution of {len(self._agents)} agents")
        
        while active_agents:
            # Create tasks to get next output from each active agent
            pending_tasks = {}
            
            for agent_idx in list(active_agents):
                agent_iter = agent_iterators[agent_idx][1]
                task = asyncio.create_task(agent_iter.__anext__())
                pending_tasks[task] = agent_idx
            
            if not pending_tasks:
                break
            
            # Wait for at least one agent to produce output
            done, pending = await asyncio.wait(
                pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for task in done:
                agent_idx = pending_tasks[task]
                agent_id = self._agent_ids[agent_idx]
                
                try:
                    output = await task
                    yield SubagentOutput(
                        agent_index=agent_idx,
                        agent_id=agent_id,
                        output=output,
                        is_final=False
                    )
                    logger.debug(f"Agent {agent_idx} produced output: {type(output).__name__}")
                    
                except StopAsyncIteration:
                    # Agent finished
                    active_agents.remove(agent_idx)
                    yield SubagentOutput(
                        agent_index=agent_idx,
                        agent_id=agent_id,
                        output=None,
                        is_final=True
                    )
                    logger.info(f"Agent {agent_idx} ({agent_id}) completed")
                    
                except Exception as e:
                    # Agent errored
                    active_agents.remove(agent_idx)
                    yield SubagentOutput(
                        agent_index=agent_idx,
                        agent_id=agent_id,
                        output=None,
                        is_final=True,
                        error=str(e)
                    )
                    logger.error(f"Agent {agent_idx} ({agent_id}) failed: {e}")
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
        
        logger.info("All agents completed")


async def spawn_parallel_agents(
    client: openai.AsyncOpenAI,
    agent_specs: list[tuple[str, str, list[Callable[..., Any]]]],
    model: str = "gpt-4",
    max_tokens: int = 8192,
    reasoning_max_tokens: Optional[int] = None,
    max_iterations: int = 50
) -> AsyncIterator[SubagentOutput]:
    """Spawn multiple subagents in parallel and yield their outputs.
    
    This is the main function for spawning parallel subagents. Each subagent runs
    independently and their outputs are yielded as they become available.
    
    Args:
        client: OpenAI async client
        agent_specs: List of (system_prompt, prompt, tools) tuples defining each agent
        model: Model to use for all agents
        max_tokens: Maximum tokens per response
        reasoning_max_tokens: Maximum tokens for reasoning (for reasoning models)
        max_iterations: Maximum iterations per agent
    
    Yields:
        SubagentOutput objects containing outputs from each agent as they execute
        
    Example:
        ```python
        import openai
        from toololo.lib.subagent import spawn_parallel_agents
        from toololo.lib import shell_command
        
        client = openai.AsyncOpenAI()
        
        specs = [
            ("You are a file analyzer", "Analyze the file structure", [list_directory]),
            ("You are a code reviewer", "Review Python files", [read_file, shell_command]),
        ]
        
        async for result in spawn_parallel_agents(client, specs):
            print(f"Agent {result.agent_index}: {result.output}")
        ```
    """
    manager = ParallelSubagents(
        client=client,
        model=model,
        max_tokens=max_tokens,
        reasoning_max_tokens=reasoning_max_tokens,
        max_iterations=max_iterations
    )
    
    # Convert agent_specs to the new format
    agent_prompts = [prompt for system_prompt, prompt, tools in agent_specs]
    system_prompts = [system_prompt for system_prompt, prompt, tools in agent_specs]
    
    # Use the first spec's tools (assuming all have same tools for backwards compatibility)
    if agent_specs:
        manager.tools = agent_specs[0][2]
    
    async for output in manager.spawn_agents(agent_prompts, system_prompts):
        yield output
