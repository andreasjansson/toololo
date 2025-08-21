import asyncio
import inspect
import json
import logging
import traceback
from typing import Any, AsyncIterator, Callable, Optional, cast

import openai

from .function import function_to_jsonschema, hashed_function_name, make_compatible
from .types import Output, TextContent, ThinkingContent, ToolResult, ToolUseContent

logger = logging.getLogger(__name__)


class Run:
    def __init__(
        self,
        client: openai.AsyncOpenAI,
        messages: list | str,
        model: str,
        tools: list[Callable[..., Any]],
        system_prompt: str = "",
        max_tokens=8192,
        reasoning_max_tokens: int = None,
        max_iterations=50,
    ):
        logger.info(f"Initializing Run with model={model}, {len(tools)} tools, max_iterations={max_iterations}")
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_max_tokens = reasoning_max_tokens
        self.max_iterations = max_iterations

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

        self.system_prompt = system_prompt
        logger.debug(f"System prompt: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"System prompt: {system_prompt}")

        self.pending_user_messages = []
        self.iteration = 0

        self.initialized = False
        self._generator: Optional[AsyncIterator[Output]] = None
        
        logger.debug(f"Tool functions: {[func.__name__ for func in tools]}")

    async def initialize(self) -> None:
        if self.initialized:
            logger.debug("Already initialized, skipping")
            return

        logger.info(f"Initializing tool schemas for {len(self.compatible_tools)} functions")
        try:
            # Execute all function_to_jsonschema calls in parallel
            tasks = [
                function_to_jsonschema(self.client, self.model, func)
                for func in self.compatible_tools
            ]
            self.tool_schemas = await asyncio.gather(*tasks)
            logger.info(f"Successfully generated {len(self.tool_schemas)} tool schemas")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize tool schemas: {e}")
            logger.error(f"Exception details: {traceback.format_exc()}")
            raise

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

            # Get response from model
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                try:
                    # Prepare messages with system prompt
                    messages = []
                    if self.system_prompt:
                        messages.append({"role": "system", "content": self.system_prompt})
                    messages.extend(self.messages)
                    
                    # Prepare request parameters
                    params = {
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        "messages": messages,
                    }
                    
                    if self.tool_schemas:
                        params["tools"] = self.tool_schemas
                        
                    if self.reasoning_max_tokens:
                        params["reasoning"] = {"max_tokens": self.reasoning_max_tokens}
                    
                    response = await self.client.chat.completions.create(**params)
                    break
                except openai.APIStatusError:
                    attempt += 1
                    await asyncio.sleep(30)
                    if attempt >= max_attempts:
                        return

            # Process the response
            message = response.choices[0].message
            assistant_message_content = message.content
            
            # Yield reasoning content if present
            if hasattr(message, 'reasoning') and message.reasoning:
                yield ThinkingContent(message.reasoning)
            
            # Yield text content if present
            if assistant_message_content:
                yield TextContent(assistant_message_content)

            # Find all tool calls for parallel processing
            tool_use_tasks = []
            tool_use_contents = []

            # Process tool calls if present
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)

                    # Yield the tool use
                    tool_content = ToolUseContent(func_name, func_args)
                    yield tool_content
                    tool_use_contents.append((tool_call, tool_content))

                    # Create task for parallel execution
                    if func_name in self.function_map:
                        func = self.function_map[func_name]
                        original_func = self.original_function_map[func_name]
                        task = self._execute_function(func, **func_args)
                        tool_use_tasks.append((tool_call, task, original_func, True))
                    else:
                        error_msg = f"Invalid tool: {func_name}. Valid available tools are: {', '.join(self.function_map.keys())}"
                        tool_use_tasks.append((tool_call, error_msg, None, False))

            # Execute all tool calls in parallel if there are any
            if tool_use_tasks:
                # Wait for all tasks to complete (or error)
                tool_results = []
                for tool_call, task_or_error, original_func, is_task in tool_use_tasks:
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

                    # Prepare the tool result for the model
                    tool_result = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": result_content,
                    }

                    tool_results.append(tool_result)

            # If no tool calls, we're done
            else:
                assistant_msg = {"role": "assistant", "content": assistant_message_content}
                if hasattr(message, 'reasoning') and message.reasoning:
                    assistant_msg["reasoning"] = message.reasoning
                if message.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                self.messages.append(assistant_msg)
                return

            # Add the messages for the next iteration
            assistant_msg = {"role": "assistant", "content": assistant_message_content}
            if hasattr(message, 'reasoning') and message.reasoning:
                assistant_msg["reasoning"] = message.reasoning
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function", 
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            
            self.messages.append(assistant_msg)
            self.messages.extend(tool_results)

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
