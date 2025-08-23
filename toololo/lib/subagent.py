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
    agent_iterators = [(i, agent.__aiter__()) for i, agent in enumerate(agents)]
    active_agents = set(range(len(agents)))
    
    logger.info(f"Starting parallel execution of {len(agents)} agents")
    
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
            
            try:
                output = await task
                yield agent_idx, output
                logger.debug(f"Agent {agent_idx} produced output: {type(output).__name__}")
                
            except StopAsyncIteration:
                # Agent finished
                active_agents.remove(agent_idx)
                logger.info(f"Agent {agent_idx} completed")
                
            except Exception as e:
                # Agent errored
                active_agents.remove(agent_idx)
                logger.error(f"Agent {agent_idx} failed: {e}")
        
        # Cancel pending tasks
        for task in pending:
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
        self._agents: list[Run] = []
        self._agent_ids: list[str] = []
    
    async def spawn_agents(
        self,
        agent_prompts: list[str] | list[tuple[str, str]],
        system_prompt: str | list[str] = ""
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
        self._agents = []
        self._agent_ids = []
        
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
            
            agent_id = f"agent_{i}_{hash((sys_prompt, prompt))}"
            self._agent_ids.append(agent_id)
            
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
            self._agents.append(agent)
            logger.info(f"Created subagent {i} ({agent_id}) with {len(self.tools)} tools")
        
        # Run all agents in parallel and collect final messages
        final_messages = []
        async for output in self._run_agents_parallel():
            if output.is_final:
                if not output.error:
                    logger.info(f"Agent {output.agent_index} completed successfully")
                else:
                    logger.error(f"Agent {output.agent_index} failed: {output.error}")
            else:
                # Collect final assistant messages (TextContent outputs)
                if isinstance(output.output, TextContent):
                    final_messages.append(output.output.content)
                    logger.debug(f"Agent {output.agent_index} produced text: {output.output.content[:100]}...")
        
        logger.info(f"Collected {len(final_messages)} final messages from spawned agents")
        return final_messages
    
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



