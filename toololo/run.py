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
        logger.info("Starting output generation")
        try:
            await self.initialize()
            logger.info(f"Beginning iteration loop (max {self.max_iterations})")
            for self.iteration in range(self.max_iterations):
                logger.debug(f"Starting iteration {self.iteration}")
                # Process any pending user messages
                if self.pending_user_messages:
                    for message in self.pending_user_messages:
                        self.messages.append({"role": "user", "content": message})
                    self.pending_user_messages = []

                # Get response from model
                logger.debug(f"Making API request for iteration {self.iteration}")
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
                            logger.debug(f"Including {len(self.tool_schemas)} tools in request")
                            
                        if self.reasoning_max_tokens:
                            params["reasoning"] = {"max_tokens": self.reasoning_max_tokens}
                            logger.debug(f"Including reasoning with max_tokens={self.reasoning_max_tokens}")
                        
                        logger.info(f"Making API call to {self.model} with {len(messages)} messages")
                        response = await self.client.chat.completions.create(**params)
                        logger.info("API call successful")
                        break
                    except openai.APIStatusError as e:
                        logger.warning(f"API status error on attempt {attempt + 1}/{max_attempts}: {e}")
                        logger.warning(f"Status code: {e.status_code}, Response: {e.response}")
                        attempt += 1
                        if attempt < max_attempts:
                            logger.info(f"Sleeping 30 seconds before retry...")
                            await asyncio.sleep(30)
                        else:
                            logger.error("Max API attempts reached, giving up")
                            return
                    except Exception as e:
                        logger.error(f"Unexpected error during API call: {e}")
                        logger.error(f"Exception details: {traceback.format_exc()}")
                        raise

                # Process the response
                message = response.choices[0].message
                assistant_message_content = message.content
                
                logger.debug(f"Iteration {self.iteration} - Content: {assistant_message_content[:100] if assistant_message_content else None}")
                logger.debug(f"Iteration {self.iteration} - Tool calls: {bool(message.tool_calls)}")
                logger.debug(f"Iteration {self.iteration} - Finish reason: {response.choices[0].finish_reason}")
                
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
                        
                        # Handle empty or invalid function arguments
                        try:
                            if tool_call.function.arguments.strip():
                                func_args = json.loads(tool_call.function.arguments)
                            else:
                                logger.warning(f"Empty function arguments for {func_name}, using empty dict")
                                func_args = {}
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in function arguments for {func_name}: {tool_call.function.arguments}")
                            logger.error(f"JSON decode error: {e}")
                            func_args = {}

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
                    logger.info(f"Executing {len(tool_use_tasks)} tool calls")
                    # Wait for all tasks to complete (or error)
                    tool_results = []
                    for i, (tool_call, task_or_error, original_func, is_task) in enumerate(tool_use_tasks):
                        func_name = original_func.__name__ if original_func else "unknown"
                        logger.debug(f"Processing tool call {i+1}/{len(tool_use_tasks)}: {func_name}")
                        
                        if is_task:
                            try:
                                # Execute the task
                                logger.debug(f"Executing function {func_name}")
                                result = await task_or_error
                                result_content = json.dumps(result)
                                success = True
                                logger.info(f"Tool call {func_name} succeeded")
                            except Exception as e:
                                logger.error(f"Tool call {func_name} failed: {e}")
                                logger.error(f"Tool call exception: {traceback.format_exc()}")
                                result_content = "".join(traceback.format_exception(e))
                                success = False
                        else:
                            # This is already an error message
                            logger.warning(f"Tool call failed with error: {task_or_error}")
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
                    
                    logger.info(f"Completed all {len(tool_use_tasks)} tool calls")

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
                if tool_use_tasks:  # Only extend if there are tool results
                    self.messages.extend(tool_results)
                
                logger.debug(f"Iteration {self.iteration} completed, continuing to next iteration")
                
                # If no tool calls, we're done
                if not tool_use_tasks:
                    logger.info(f"No tool calls in response, task completed after {self.iteration + 1} iterations")
                    return
            
        except Exception as e:
            logger.error(f"Error in output generation: {e}")
            logger.error(f"Generation error details: {traceback.format_exc()}")
            raise

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
