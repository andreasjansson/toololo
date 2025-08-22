#!/usr/bin/env python3
"""
Simple demo showing subagent concept without OpenAI API calls.

This demonstrates the parallel execution and tool binding concepts
without needing real OpenAI API calls or schema generation.
"""

import asyncio
import tempfile
from pathlib import Path

from toololo.lib.subagent import ParallelSubagents
from toololo.lib.files import write_file, list_directory
from toololo.lib.example_tools import (
    analyze_code_file, find_files_with_pattern, count_lines_in_files
)


def demo_tool_binding():
    """Demonstrate tool binding with client parameter."""
    print("🔧 Tool Binding Demo")
    print("-" * 30)
    
    # Create a mock client
    class MockClient:
        def __init__(self, name):
            self.name = name
        
        def process(self, data):
            return f"Processed {data} with {self.name}"
    
    client = MockClient("DemoClient")
    
    # Create a tool that needs a client
    def tool_with_client(text: str, multiplier: int = 1, client=None) -> str:
        """Demo tool that uses a client parameter."""
        if client:
            result = client.process(text)
            return f"{result} (x{multiplier})"
        return f"No client: {text} (x{multiplier})"
    
    # Create a simple tool
    def simple_tool(text: str) -> str:
        """Simple tool without client dependency."""
        return f"Simple processing: {text.upper()}"
    
    # Test binding
    manager = ParallelSubagents(client=client)
    bound_tools = manager._bind_client_to_tools([tool_with_client, simple_tool])
    
    print("✅ Original tools:")
    print(f"   tool_with_client: {tool_with_client.__name__}")
    print(f"   simple_tool: {simple_tool.__name__}")
    
    print("✅ Testing bound tools:")
    result1 = bound_tools[0]("hello", multiplier=3)
    result2 = bound_tools[1]("world")
    
    print(f"   Bound tool result: {result1}")
    print(f"   Simple tool result: {result2}")
    
    return bound_tools


def demo_file_tools():
    """Demonstrate the example tools working with files."""
    print("\n📁 File Tools Demo")
    print("-" * 30)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        
        # Create demo files
        files_to_create = {
            "main.py": """#!/usr/bin/env python3
def main():
    print("Hello World")

if __name__ == "__main__":
    main()
""",
            "utils.py": """def helper():
    return "help"

class Utils:
    def process(self):
        pass
""",
            "README.md": "# Demo Project\nThis is a test project.",
            "config.json": '{"version": "1.0"}'
        }
        
        print(f"📝 Creating {len(files_to_create)} demo files...")
        for filename, content in files_to_create.items():
            write_file(str(project_dir / filename), content)
        
        # Test file tools
        print("\n🔍 Testing file analysis tools:")
        
        # Test 1: Find Python files
        py_files_result = find_files_with_pattern(str(project_dir), "*.py")
        print(f"   Python files found: {py_files_result}")
        
        # Test 2: Analyze a code file
        main_py_path = project_dir / "main.py"
        code_analysis = analyze_code_file(str(main_py_path))
        print(f"   Code analysis: {code_analysis}")
        
        # Test 3: Count lines in Python files
        line_count = count_lines_in_files(str(project_dir), "*.py")
        print(f"   Line counts: {line_count}")
        
        # Test 4: List directory
        directory_listing = list_directory(str(project_dir))
        print(f"   Directory listing:\n{directory_listing}")
        
        print("✅ All file tools working correctly!")


async def demo_parallel_concept():
    """Demonstrate the parallel execution concept with simple tasks."""
    print("\n⚡ Parallel Execution Concept Demo")
    print("-" * 30)
    
    # Simulate agent tasks without OpenAI
    async def simulate_agent_work(agent_id: str, task_name: str, duration: float):
        """Simulate agent doing work."""
        print(f"   🤖 Agent {agent_id} starting: {task_name}")
        await asyncio.sleep(duration)
        print(f"   ✅ Agent {agent_id} completed: {task_name}")
        return f"Result from {agent_id}: {task_name}"
    
    # Create multiple agent tasks
    agent_tasks = [
        simulate_agent_work("A", "File Structure Analysis", 0.5),
        simulate_agent_work("B", "Code Quality Check", 0.3), 
        simulate_agent_work("C", "Documentation Review", 0.4),
    ]
    
    print("🚀 Starting parallel agents...")
    start_time = asyncio.get_event_loop().time()
    
    # Run all agents in parallel
    results = await asyncio.gather(*agent_tasks)
    
    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time
    
    print(f"🎉 All agents completed in {duration:.2f}s")
    print("📊 Results:")
    for i, result in enumerate(results):
        print(f"   {i+1}. {result}")
    
    return results


def main():
    """Main demo function."""
    print("🤖 Simple Subagent Demo")
    print("=" * 50)
    print("This demo shows the core concepts of the subagent system:")
    print("1. Tool binding with client parameters")
    print("2. File analysis tools")
    print("3. Parallel execution concepts")
    print()
    
    # Demo 1: Tool binding
    bound_tools = demo_tool_binding()
    
    # Demo 2: File tools
    demo_file_tools()
    
    # Demo 3: Parallel concept
    asyncio.run(demo_parallel_concept())
    
    print("\n" + "=" * 50)
    print("✨ Demo Summary:")
    print("   • Tool binding works correctly")
    print("   • File analysis tools are functional")
    print("   • Parallel execution concept demonstrated")
    print("   • Ready for real OpenAI integration!")
    print()
    print("💡 To use with real OpenAI:")
    print("   1. Set up OpenAI API key")
    print("   2. Replace mock client with openai.AsyncOpenAI()")
    print("   3. Use spawn_parallel_agents() function")


if __name__ == "__main__":
    main()
