# toololo.lib.subagent - Parallel AI Agent Execution

The `toololo.lib.subagent` module enables spawning multiple AI agents that run in parallel, each with their own specialization and tools. This is useful for complex tasks that can be broken down into parallel subtasks handled by specialized agents.

## Key Features

- **Parallel Execution**: Multiple agents run simultaneously, not sequentially
- **Tool Binding**: Automatically binds OpenAI client to tools that need it using `functools.partial`
- **Streaming Output**: Yields outputs from agents as they become available
- **Agent Isolation**: Each agent has its own system prompt, user prompt, and tool set
- **Error Handling**: Gracefully handles agent failures without stopping others

## Core Functions

### `spawn_parallel_agents()`

The main function for creating and running parallel agents:

```python
async def spawn_parallel_agents(
    client: openai.AsyncOpenAI,
    agent_specs: List[Tuple[str, str, List[Callable[..., Any]]]],
    model: str = "gpt-4",
    max_tokens: int = 8192,
    reasoning_max_tokens: Optional[int] = None,
    max_iterations: int = 50
) -> AsyncIterator[SubagentOutput]
```

**Parameters:**
- `client`: OpenAI async client instance
- `agent_specs`: List of `(system_prompt, user_prompt, tools)` tuples
- `model`: Model to use for all agents
- `max_tokens`: Maximum tokens per response
- `reasoning_max_tokens`: Max reasoning tokens (for reasoning models)
- `max_iterations`: Maximum iterations per agent

**Yields:**
- `SubagentOutput` objects containing agent results as they execute

## Data Types

### `SubagentOutput`

```python
@dataclass
class SubagentOutput:
    agent_index: int      # Which agent (0-based)
    agent_id: str         # Unique agent identifier
    output: Output        # The actual output from toololo
    is_final: bool        # True when agent completes
    error: Optional[str]  # Error message if agent failed
```

## Usage Examples

### Basic Parallel Analysis

```python
import asyncio
import openai
from toololo.lib.subagent import spawn_parallel_agents
from toololo.lib.files import read_file, list_directory
from toololo.lib.shell import shell_command

async def analyze_project():
    client = openai.AsyncOpenAI(api_key="your-key")
    
    # Define specialized agents
    agent_specs = [
        (
            "You are a file structure analyst.",
            "Analyze the project structure in /path/to/project",
            [list_directory, shell_command]
        ),
        (
            "You are a code quality expert.",
            "Review Python code quality in /path/to/project", 
            [read_file, shell_command]
        ),
        (
            "You are a documentation reviewer.",
            "Evaluate project documentation completeness",
            [read_file, list_directory]
        )
    ]
    
    # Run agents in parallel
    async for result in spawn_parallel_agents(client, agent_specs):
        if result.is_final:
            if result.error:
                print(f"Agent {result.agent_index} failed: {result.error}")
            else:
                print(f"Agent {result.agent_index} completed successfully")
        else:
            print(f"Agent {result.agent_index}: {type(result.output).__name__}")

# Run the analysis
asyncio.run(analyze_project())
```

### Tools with Client Dependencies

The subagent system automatically binds the OpenAI client to tools that have a `client` parameter:

```python
async def ai_analysis_tool(text: str, analysis_type: str, client: openai.AsyncOpenAI) -> str:
    """Tool that uses OpenAI API for analysis."""
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Analyze this text: {text}"}],
        max_tokens=200
    )
    return response.choices[0].message.content

# This tool will automatically receive the client when used by agents
agent_specs = [
    (
        "You are an AI analysis expert.",
        "Analyze the given text for sentiment and themes.",
        [ai_analysis_tool]  # Client will be bound automatically
    )
]
```

### Collaborative Multi-Agent Workflow

```python
async def collaborative_analysis(project_path: str):
    client = openai.AsyncOpenAI(api_key="your-key")
    
    # Agents with different specializations
    agents = [
        # Structure analyst
        (
            "You specialize in analyzing project structure and organization.",
            f"Analyze the directory structure of {project_path}",
            [list_directory, find_files_with_pattern]
        ),
        # Code analyst  
        (
            "You specialize in code quality and metrics analysis.",
            f"Analyze Python code quality in {project_path}",
            [analyze_code_file, count_lines_in_files]
        ),
        # Security analyst
        (
            "You specialize in security and dependency analysis.",
            f"Review {project_path} for security issues",
            [read_file, shell_command]
        ),
        # Documentation analyst
        (
            "You specialize in documentation quality assessment.",
            f"Evaluate documentation in {project_path}",
            [read_file, find_files_with_pattern, ai_analysis_tool]
        )
    ]
    
    # Collect results from all agents
    results = {"structure": [], "code": [], "security": [], "docs": []}
    agent_names = ["structure", "code", "security", "docs"]
    
    async for output in spawn_parallel_agents(client, agents, max_iterations=10):
        category = agent_names[output.agent_index]
        
        if output.is_final:
            print(f"{category.title()} analysis completed")
        else:
            results[category].append(output.output)
    
    # Generate final collaborative report
    print("🎉 All agents completed. Generating final report...")
    return results
```

## Advanced Features

### Custom Agent Manager

For more control, use the `ParallelSubagents` class directly:

```python
from toololo.lib.subagent import ParallelSubagents

async def custom_agent_workflow():
    client = openai.AsyncOpenAI(api_key="your-key")
    
    manager = ParallelSubagents(
        client=client,
        model="gpt-4",
        max_tokens=4096,
        max_iterations=20
    )
    
    agent_specs = [
        ("System prompt 1", "User prompt 1", [tool1, tool2]),
        ("System prompt 2", "User prompt 2", [tool3, tool4]),
    ]
    
    async for output in manager.spawn_agents(agent_specs):
        # Custom handling logic
        handle_agent_output(output)
```

### Error Handling

```python
async def robust_parallel_analysis():
    agent_specs = [
        # Agent configurations...
    ]
    
    completed_agents = set()
    failed_agents = set()
    
    async for result in spawn_parallel_agents(client, agent_specs):
        if result.is_final:
            if result.error:
                print(f"❌ Agent {result.agent_index} failed: {result.error}")
                failed_agents.add(result.agent_index)
            else:
                print(f"✅ Agent {result.agent_index} completed")
                completed_agents.add(result.agent_index)
        
        # Continue until all agents finish (success or failure)
        if len(completed_agents | failed_agents) == len(agent_specs):
            break
    
    print(f"Summary: {len(completed_agents)} succeeded, {len(failed_agents)} failed")
```

## Tool Requirements

Tools used by subagents should:

1. **Be async-compatible**: Either async functions or sync functions (will be wrapped)
2. **Have proper type hints**: For better error handling and documentation
3. **Handle errors gracefully**: Return error messages rather than raising exceptions when possible
4. **Accept client parameter if needed**: For tools that need OpenAI API access

```python
# Good tool example
async def good_tool(input_text: str, options: dict = None, client: openai.AsyncOpenAI = None) -> str:
    """Well-designed tool with proper typing and error handling."""
    try:
        if client and options.get('use_ai'):
            # Use AI for processing
            response = await client.chat.completions.create(...)
            return response.choices[0].message.content
        else:
            # Fallback processing
            return f"Processed: {input_text}"
    except Exception as e:
        return f"Error in tool: {str(e)}"
```

## Performance Considerations

- **Parallel Speedup**: N agents can potentially provide N× speedup for independent tasks
- **Rate Limits**: Be aware of OpenAI API rate limits when running many agents
- **Memory Usage**: Each agent maintains its own conversation history
- **Timeout Handling**: Set appropriate `max_iterations` to prevent runaway agents

## Best Practices

1. **Specialized Agents**: Give each agent a clear, focused role
2. **Balanced Workload**: Distribute work evenly across agents
3. **Error Recovery**: Handle individual agent failures gracefully
4. **Resource Management**: Monitor API usage and costs
5. **Result Aggregation**: Plan how to combine results from multiple agents

## See Also

- **Examples**: `examples/simple_subagent_demo.py` - Basic concepts demo
- **Examples**: `examples/subagent_demo.py` - Full OpenAI integration example  
- **Tests**: `test-integration/test_subagent.py` - Comprehensive test suite
- **Tools**: `toololo.lib.example_tools` - Example tools for agents
