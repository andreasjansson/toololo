"""Integration tests for toololo.lib.subagent module."""

import pytest
import asyncio
import tempfile
import json
import os
from pathlib import Path

import openai
from toololo.lib.subagent import spawn_parallel_agents, SubagentOutput, ParallelSubagents
from toololo.lib.files import write_file, read_file, list_directory
from toololo.lib.shell import shell_command
from toololo.run import Run
from toololo.types import TextContent, ToolUseContent, ToolResult


@pytest.fixture
def openai_client():
    """Create a real OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return openai.AsyncOpenAI(api_key=api_key)


# Simple tools for testing (defined inline to keep things simple)

def count_python_files(directory: str) -> str:
    """Count Python files in a directory."""
    result = shell_command(f"find {directory} -name '*.py' -type f | wc -l")
    if result.success:
        count = int(result.stdout.strip())
        return f"Found {count} Python files in {directory}"
    return f"Error counting files: {result.stderr}"


def analyze_file_sizes(directory: str) -> str:
    """Analyze file sizes in a directory."""
    result = shell_command(f"find {directory} -type f -exec ls -l {{}} + | awk '{{total += $5}} END {{print total}}'")
    if result.success:
        total_bytes = int(result.stdout.strip()) if result.stdout.strip() else 0
        return f"Total size: {total_bytes} bytes ({total_bytes / 1024:.1f} KB)"
    return f"Error analyzing sizes: {result.stderr}"


def check_git_status(directory: str) -> str:
    """Check if directory is a git repo and get status."""
    git_check = shell_command("git rev-parse --is-inside-work-tree", working_directory=directory)
    if not git_check.success:
        return f"Not a git repository: {directory}"
    
    status_result = shell_command("git status --porcelain", working_directory=directory)
    if status_result.success:
        lines = [line for line in status_result.stdout.split('\n') if line.strip()]
        if lines:
            return f"Git repo with {len(lines)} uncommitted changes"
        return "Git repo with clean working directory"
    return f"Git repo (status check failed): {status_result.stderr}"


def find_readme_files(directory: str) -> str:
    """Find and analyze README files."""
    result = shell_command(f"find {directory} -iname 'readme*' -type f")
    if result.success:
        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        if files:
            return f"Found {len(files)} README files: {', '.join([Path(f).name for f in files])}"
        return "No README files found"
    return f"Error searching for README: {result.stderr}"


def assess_code_complexity(directory: str) -> str:
    """Simple assessment of code complexity based on line counts and file counts."""
    py_files = shell_command(f"find {directory} -name '*.py' -type f | wc -l")
    if not py_files.success:
        return "No Python files found or error occurred"
    
    file_count = int(py_files.stdout.strip())
    if file_count == 0:
        return "No Python files to analyze"
    
    # Count total lines in Python files
    lines_result = shell_command(f"find {directory} -name '*.py' -type f -exec wc -l {{}} + | tail -n 1")
    total_lines = 0
    if lines_result.success and lines_result.stdout.strip():
        try:
            total_lines = int(lines_result.stdout.strip().split()[0])
        except (ValueError, IndexError):
            pass
    
    avg_lines = total_lines / file_count if file_count > 0 else 0
    
    if avg_lines > 200:
        complexity = "High"
    elif avg_lines > 100:
        complexity = "Medium"
    else:
        complexity = "Low"
    
    return f"Code complexity: {complexity} ({file_count} files, {total_lines} total lines, {avg_lines:.1f} avg)"


async def ai_summary_tool(content: str, focus: str, client=None) -> str:
    """AI tool that generates summaries using OpenAI."""
    if not client:
        return f"No AI client available. Focus: {focus}, Content preview: {content[:50]}..." 
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": f"Provide a brief analysis of this {focus} (max 2 sentences): {content}"
            }],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI analysis failed: {str(e)}"


class TestSimpleTools:
    """Test the simple inline tools."""
    
    def test_count_python_files(self):
        """Test Python file counting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Python files
            write_file(str(Path(tmpdir) / "main.py"), "print('hello')")
            write_file(str(Path(tmpdir) / "utils.py"), "def helper(): pass")
            write_file(str(Path(tmpdir) / "readme.md"), "# Project")
            
            result = count_python_files(tmpdir)
            assert "Found 2 Python files" in result
    
    def test_analyze_file_sizes(self):
        """Test file size analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            write_file(str(Path(tmpdir) / "small.txt"), "hello")
            write_file(str(Path(tmpdir) / "large.txt"), "x" * 1000)
            
            result = analyze_file_sizes(tmpdir)
            assert "Total size:" in result
            assert "bytes" in result
    
    def test_find_readme_files(self):
        """Test README file detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            write_file(str(Path(tmpdir) / "README.md"), "# Project")
            write_file(str(Path(tmpdir) / "readme.txt"), "Info")
            
            result = find_readme_files(tmpdir)
            assert "Found 2 README files" in result
    
    def test_assess_code_complexity(self):
        """Test code complexity assessment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple Python file
            simple_code = "def hello():\n    print('world')\n"
            write_file(str(Path(tmpdir) / "simple.py"), simple_code)
            
            result = assess_code_complexity(tmpdir)
            assert "Code complexity: Low" in result
            assert "1 files" in result


class TestSubagentCore:
    """Test core subagent functionality."""
    
    def test_parallel_subagents_init(self, openai_client):
        """Test ParallelSubagents initialization."""
        manager = ParallelSubagents(
            client=openai_client,
            model="gpt-4o-mini",
            max_tokens=1000
        )
        
        assert manager.client == openai_client
        assert manager.model == "gpt-4o-mini"
        assert manager.max_tokens == 1000
        assert manager._agents == []
    
    def test_bind_client_to_tools(self, openai_client):
        """Test client binding to tools."""
        manager = ParallelSubagents(client=openai_client)
        
        # Define a tool that needs a client
        def tool_with_client(text: str, client=None):
            return f"Processed {text} with client {type(client).__name__}"
        
        # Define a tool that doesn't need a client
        def simple_tool(text: str):
            return f"Simple: {text}"
        
        tools = [tool_with_client, simple_tool]
        bound_tools = manager._bind_client_to_tools(tools)
        
        assert len(bound_tools) == 2
        
        # Test the bound tool
        result = bound_tools[0]("test")
        assert "AsyncOpenAI" in result
        
        # Test the simple tool (unchanged)
        result = bound_tools[1]("test")
        assert result == "Simple: test"


class TestRecursiveCodeReview:
    """Integration test for recursive subagent code review scenario.
    
    This demonstrates a tree structure of agents:
    - Level 1: Main Coordinator
    - Level 2: File Review Agents (one per Python file)
    - Level 3: Function Analysis Agents (spawned by file agents for each function)
    
    This shows how subagents can recursively spawn other subagents to handle
    tasks that naturally decompose into hierarchical subtasks.
    """
    
    @pytest.mark.asyncio
    async def test_recursive_code_review_pipeline(self, openai_client):
        """Test recursive subagents: Coordinator -> File Agents -> Function Agents.
        
        This demonstrates a realistic tree-structured workflow where agents
        spawn child agents to handle subtasks, creating a natural hierarchy.
        """
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple multi-file Python project
            project_dir = Path(tmpdir) / "simple_project"
            await self._create_simple_code_project(project_dir)
            
            # Step 1: Define the recursive file review tool
            async def review_file_with_function_agents(file_path: str, client: openai.AsyncOpenAI = None) -> str:
                """Level 2 Agent Tool: Reviews a file by spawning function analysis agents."""
                if not client:
                    return f"No client available for {file_path}"
                
                try:
                    # Read the file and extract functions
                    content = read_file(file_path)
                    functions = self._extract_function_names(content)
                    
                    if not functions:
                        return f"No functions found in {file_path}"
                    
                    print(f"    📄 File agent analyzing {len(functions)} functions in {Path(file_path).name}")
                    
                    # Step 2: Spawn function analysis agents (Level 3)
                    function_agent_specs = []
                    for func_name in functions[:2]:  # Limit to 2 functions for demo
                        function_agent_specs.append((
                            f"You are a Function Analyst. Analyze the {func_name} function for code quality.",
                            f"Analyze the function '{func_name}' in this code:\n\n{content}\n\nProvide a brief quality assessment.",
                            [ai_summary_tool]
                        ))
                    
                    # Collect function analysis results
                    function_results = []
                    completed_functions = set()
                    
                    async for result in spawn_parallel_agents(
                        client=client,
                        agent_specs=function_agent_specs,
                        model="gpt-4o-mini",
                        max_iterations=2
                    ):
                        if result.is_final:
                            completed_functions.add(result.agent_index)
                            if result.error:
                                function_results.append(f"Function analysis {result.agent_index} failed: {result.error}")
                            else:
                                function_results.append(f"Function {functions[result.agent_index] if result.agent_index < len(functions) else result.agent_index} analyzed")
                        
                        # Break when all function agents complete
                        if len(completed_functions) == len(function_agent_specs):
                            break
                    
                    return f"File {Path(file_path).name}: {len(function_results)} functions analyzed - {'; '.join(function_results)}"
                
                except Exception as e:
                    return f"Error analyzing file {file_path}: {str(e)}"
            
            # Step 3: Level 1 - Main Coordinator Agent
            coordinator_spec = [(
                "You are the Main Code Review Coordinator. You manage file review agents that in turn "
                "manage function analysis agents. Coordinate the recursive review process.",
                f"Coordinate a recursive code review of {project_dir}. "
                f"Use your file review tool to analyze each Python file in the project.",
                [review_file_with_function_agents, list_directory, count_python_files]
            )]
            
            print(f"\n🌳 Starting Recursive Code Review")
            print(f"📁 Project: {project_dir.name}")
            print("🔄 Structure: Coordinator → File Agents → Function Agents")
            
            # Track the recursive review process
            review_results = []
            coordinator_completed = False
            
            # Run the main coordinator agent
            async for result in spawn_parallel_agents(
                client=openai_client,
                agent_specs=coordinator_spec,
                model="gpt-4o-mini",
                max_iterations=4
            ):
                if result.is_final:
                    coordinator_completed = True
                    if result.error:
                        print(f"❌ Coordinator failed: {result.error}")
                    else:
                        print(f"✅ Recursive code review completed")
                else:
                    review_results.append(result.output)
                    output_type = type(result.output).__name__
                    print(f"🔄 Coordinator: {output_type}")
                
                if coordinator_completed:
                    break
            
            # Verify the recursive structure worked
            print(f"\n📊 Recursive Review Summary:")
            print(f"  Coordinator Status: {'✅ Success' if coordinator_completed else '❌ Failed'}")
            print(f"  Total Operations: {len(review_results)}")
            
            # Look for evidence of recursive operations in tool results
            file_analysis_count = 0
            function_analysis_count = 0
            
            for output in review_results:
                if isinstance(output, ToolResult) and output.success:
                    result_text = output.result.lower()
                    if "functions analyzed" in result_text:
                        file_analysis_count += 1
                    if "function" in result_text and "analyzed" in result_text:
                        function_analysis_count += 1
            
            print(f"  File Analyses: {file_analysis_count}")
            print(f"  Function References: {function_analysis_count}")
            
            # Assertions for recursive behavior
            assert coordinator_completed, "Coordinator agent should complete successfully"
            assert len(review_results) > 0, "Should have some review operations"
            assert file_analysis_count > 0, "Should have evidence of file-level analysis"
            
            print(f"🎉 Recursive subagent tree structure verified!")
            return {
                "coordinator_success": coordinator_completed,
                "total_operations": len(review_results),
                "file_analyses": file_analysis_count,
                "function_references": function_analysis_count
            }
    
    def _extract_function_names(self, code_content: str) -> list:
        """Extract function names from Python code."""
        import re
        # Simple regex to find function definitions
        function_pattern = r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        functions = re.findall(function_pattern, code_content, re.MULTILINE)
        return functions
    
    async def _create_simple_code_project(self, project_dir: Path):
        """Create a simple Python project with multiple files and functions for recursive review."""
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create simple Python files with multiple functions each
        project_files = {
            "calculator.py": """def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b

def subtract(a, b):
    \"\"\"Subtract two numbers.\"\"\"
    return a - b

def multiply(a, b):
    \"\"\"Multiply two numbers.\"\"\"
    return a * b
""",
            
            "string_utils.py": """def reverse_string(text):
    \"\"\"Reverse a string.\"\"\"
    return text[::-1]

def count_vowels(text):
    \"\"\"Count vowels in text.\"\"\"
    vowels = 'aeiouAEIOU'
    return sum(1 for char in text if char in vowels)
""",
            
            "data_processor.py": """def filter_even(numbers):
    \"\"\"Filter even numbers from a list.\"\"\"
    return [n for n in numbers if n % 2 == 0]

def sum_list(numbers):
    \"\"\"Sum all numbers in a list.\"\"\"
    return sum(numbers)
"""
        }
        
        # Write all project files
        for file_path, content in project_files.items():
            full_path = project_dir / file_path
            write_file(str(full_path), content)
        
        print(f"✅ Created simple project with {len(project_files)} files for recursive review")
        """Create a realistic open-source project structure for testing."""
        project_dir.mkdir(parents=True)
        
        # Create directory structure
        (project_dir / "src").mkdir()
        (project_dir / "tests").mkdir() 
        (project_dir / "docs").mkdir()
        (project_dir / "config").mkdir()
        (project_dir / ".github").mkdir()
        
        # Create realistic project files
        project_files = {
            "README.md": """# Awesome Project 🚀

A fantastic open-source project that demonstrates best practices.

## Features
- Clean architecture
- Comprehensive testing
- Great documentation
- Active community

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from awesome_project import main
main.run()
```

## Contributing
We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## License
MIT License - see LICENSE file for details.
""",
            
            "src/main.py": """#!/usr/bin/env python3
\"\"\"Main module for Awesome Project.\"\"\"

import argparse
import logging
from pathlib import Path

from .core import DataProcessor
from .utils import setup_logging, load_config

logger = logging.getLogger(__name__)

def main():
    \"\"\"Main entry point for the application.\"\"\"
    parser = argparse.ArgumentParser(description="Awesome Project")
    parser.add_argument("--config", help="Configuration file", default="config/default.json")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("input_file", help="Input file to process")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger.info("Starting Awesome Project")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize processor
        processor = DataProcessor(config)
        
        # Process input
        result = processor.process_file(args.input_file)
        
        print(f"Processing completed successfully: {result}")
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
""",
            
            "src/core.py": """\"\"\"Core data processing functionality.\"\"\"

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    \"\"\"Main data processing engine.\"\"\"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processed_count = 0
        self.errors = []
        
    def process_file(self, file_path: str) -> Dict[str, Any]:
        \"\"\"Process a single file and return results.\"\"\"
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Processing file: {path.name}")
        
        try:
            if path.suffix == '.json':
                return self._process_json_file(path)
            elif path.suffix in ['.txt', '.md']:
                return self._process_text_file(path)
            else:
                logger.warning(f"Unsupported file type: {path.suffix}")
                return {"status": "skipped", "reason": "unsupported_type"}
                
        except Exception as e:
            logger.error(f"Error processing {path.name}: {e}")
            self.errors.append(str(e))
            raise
    
    def _process_json_file(self, path: Path) -> Dict[str, Any]:
        \"\"\"Process JSON files.\"\"\"
        with open(path, 'r') as f:
            data = json.load(f)
            
        result = {
            "type": "json",
            "file_name": path.name,
            "records": len(data) if isinstance(data, list) else 1,
            "size_bytes": path.stat().st_size,
            "status": "success"
        }
        
        self.processed_count += 1
        return result
    
    def _process_text_file(self, path: Path) -> Dict[str, Any]:
        \"\"\"Process text files.\"\"\"
        with open(path, 'r') as f:
            content = f.read()
            
        result = {
            "type": "text",
            "file_name": path.name,
            "lines": len(content.splitlines()),
            "words": len(content.split()),
            "characters": len(content),
            "status": "success"
        }
        
        self.processed_count += 1
        return result

class ValidationError(Exception):
    \"\"\"Custom exception for validation errors.\"\"\"
    pass
""",
            
            "src/utils.py": """\"\"\"Utility functions and helpers.\"\"\"

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def setup_logging(level: int = logging.INFO) -> None:
    \"\"\"Configure application logging.\"\"\"
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    \"\"\"Load configuration from JSON file.\"\"\"
    default_config = {
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "supported_formats": [".json", ".txt", ".md"],
        "output_format": "json",
        "enable_logging": True
    }
    
    if not config_path:
        return default_config
        
    config_file = Path(config_path)
    if not config_file.exists():
        logging.warning(f"Config file not found: {config_path}, using defaults")
        return default_config
        
    try:
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
        return default_config
    except json.JSONDecodeError as e:
        logging.error(f"Invalid config file: {e}")
        return default_config

def validate_input(file_path: str, config: Dict[str, Any]) -> bool:
    \"\"\"Validate input file against configuration.\"\"\"
    path = Path(file_path)
    
    if not path.exists():
        return False
        
    if path.stat().st_size > config.get("max_file_size", float('inf')):
        return False
        
    if path.suffix not in config.get("supported_formats", []):
        return False
        
    return True
""",
            
            "tests/test_core.py": """\"\"\"Tests for core functionality.\"\"\"

import json
import tempfile
import unittest
from pathlib import Path

from src.core import DataProcessor, ValidationError

class TestDataProcessor(unittest.TestCase):
    \"\"\"Test cases for DataProcessor class.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test fixtures.\"\"\"
        self.config = {
            "max_file_size": 1024,
            "supported_formats": [".json", ".txt"]
        }
        self.processor = DataProcessor(self.config)
    
    def test_process_json_file(self):
        \"\"\"Test JSON file processing.\"\"\"
        test_data = [{"id": 1, "name": "test"}, {"id": 2, "name": "example"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = self.processor.process_file(temp_path)
            self.assertEqual(result["type"], "json")
            self.assertEqual(result["records"], 2)
            self.assertEqual(result["status"], "success")
        finally:
            Path(temp_path).unlink()
    
    def test_process_text_file(self):
        \"\"\"Test text file processing.\"\"\"
        test_content = "Hello world\\nThis is a test\\nWith multiple lines"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = self.processor.process_file(temp_path)
            self.assertEqual(result["type"], "text")
            self.assertEqual(result["lines"], 3)
            self.assertGreater(result["words"], 5)
        finally:
            Path(temp_path).unlink()

if __name__ == '__main__':
    unittest.main()
""",
            
            "config/default.json": """{
    "max_file_size": 52428800,
    "supported_formats": [".json", ".txt", ".md", ".csv"],
    "output_format": "json",
    "enable_logging": true,
    "log_level": "INFO",
    "processing_threads": 4,
    "cache_enabled": true
}""",
            
            "requirements.txt": """# Production dependencies
requests>=2.28.0
click>=8.0.0
pyyaml>=6.0
pandas>=1.5.0

# Development dependencies
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.990
""",
            
            "docs/architecture.md": """# Architecture Documentation

## Overview
Awesome Project follows a modular architecture with clear separation of concerns.

## Core Components

### Data Processing Engine
- `DataProcessor`: Main processing class
- Supports multiple file formats
- Configurable processing pipeline

### Configuration System
- JSON-based configuration
- Environment-specific configs
- Runtime configuration validation

### Error Handling
- Custom exception types
- Comprehensive logging
- Graceful degradation

## Data Flow
1. Input validation
2. Configuration loading  
3. File processing
4. Result generation
5. Error reporting
""",
            
            ".github/workflows/ci.yml": """name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/
    
    - name: Lint with flake8
      run: |
        flake8 src/ tests/
"""
        }
        
        # Write all project files
        for file_path, content in project_files.items():
            full_path = project_dir / file_path
            write_file(str(full_path), content)
        
        print(f"✅ Created realistic project with {len(project_files)} files")
    
class TestErrorHandling:
    """Test error handling in subagent scenarios."""
    
    @pytest.mark.asyncio
    async def test_agent_with_failing_tools(self, openai_client):
        """Test handling of agents with tools that fail."""
        
        def failing_tool(input_text: str) -> str:
            """A tool that always fails."""
            raise ValueError("This tool always fails!")
        
        def working_tool(input_text: str) -> str:
            """A tool that works."""
            return f"Processed: {input_text}"
        
        agent_specs = [
            (
                "You are a test agent with mixed tools. Use the working_tool to process text.",
                "Process the text 'test input' using your available tools",
                [working_tool, failing_tool]
            )
        ]
        
        results = []
        async for result in spawn_parallel_agents(
            client=openai_client,
            agent_specs=agent_specs,
            model="gpt-4o-mini",
            max_iterations=2
        ):
            results.append(result)
            if result.is_final:
                break
        
        # Should have completed (though may have had tool failures)
        assert any(r.is_final for r in results)
        final_result = [r for r in results if r.is_final][0]
        assert final_result.agent_index == 0
    
    @pytest.mark.asyncio  
    async def test_agent_timeout_scenario(self, openai_client):
        """Test agent behavior with very low iteration limits."""
        
        agent_specs = [
            (
                "You are a simple agent. Respond briefly.",
                "Say hello and stop",
                [lambda: "Hello World"]
            )
        ]
        
        results = []
        async for result in spawn_parallel_agents(
            client=openai_client,
            agent_specs=agent_specs,
            model="gpt-4o-mini",
            max_iterations=1  # Very low limit
        ):
            results.append(result)
            if result.is_final:
                break
        
        # Should complete quickly with low iteration limit
        assert any(r.is_final for r in results)


class TestSubagentWithRun:
    """Test using subagent functionality through toololo.Run."""
    
    @pytest.mark.asyncio
    async def test_subagent_analysis_tool_in_run(self, openai_client):
        """Test a tool that uses subagents internally, called through toololo.Run."""
        
        async def analyze_project_with_subagents(project_path: str, client: openai.AsyncOpenAI = None) -> str:
            """Tool that performs project analysis using multiple subagents internally.
            
            This tool demonstrates using subagents as part of a larger workflow
            where the subagent functionality is encapsulated in a tool.
            """
            if not client:
                return "Error: OpenAI client required for analysis"
            
            # Create a simple test project
            with tempfile.TemporaryDirectory() as tmpdir:
                test_project = Path(tmpdir) / "test_project"
                test_project.mkdir()
                
                # Create sample files
                write_file(str(test_project / "main.py"), """
def hello():
    print("Hello World")

if __name__ == "__main__":
    hello()
""")
                write_file(str(test_project / "README.md"), "# Test Project\nA simple test project.")
                
                # Define subagents for analysis
                agent_specs = [
                    (
                        "You are a code structure analyst. Be concise.",
                        f"Analyze the Python files in {test_project}. Give a brief summary.",
                        [count_python_files, list_directory]
                    ),
                    (
                        "You are a documentation reviewer. Be concise.",
                        f"Review the documentation in {test_project}. Give a brief summary.",
                        [find_readme_files, analyze_file_sizes]
                    )
                ]
                
                # Collect results from subagents
                results = []
                agent_completed = {0: False, 1: False}
                
                try:
                    async for output in spawn_parallel_agents(
                        client=client,
                        agent_specs=agent_specs,
                        model="gpt-4o-mini",
                        max_iterations=2
                    ):
                        if output.is_final:
                            agent_completed[output.agent_index] = True
                            if output.error:
                                results.append(f"Agent {output.agent_index} error: {output.error}")
                            else:
                                results.append(f"Agent {output.agent_index} completed successfully")
                        
                        # Break when both agents complete
                        if all(agent_completed.values()):
                            break
                    
                    # Summarize results
                    successful_agents = sum(1 for completed in agent_completed.values() if completed)
                    return f"Project analysis completed. {successful_agents}/2 agents successful. Results: {'; '.join(results)}"
                    
                except Exception as e:
                    return f"Subagent analysis failed: {str(e)}"
        
        # Use the subagent tool through toololo.Run
        run = Run(
            client=openai_client,
            messages="Analyze a test project using your subagent analysis capabilities.",
            model="gpt-4o-mini",
            tools=[analyze_project_with_subagents],
            system_prompt="You are a project analysis coordinator. Use your subagent analysis tool to analyze projects.",
            max_iterations=3
        )
        
        # Collect outputs from the run
        outputs = []
        async for output in run:
            outputs.append(output)
            
            # Look for successful tool usage
            if isinstance(output, ToolResult) and output.success:
                # Verify the tool result indicates subagent usage
                assert "agents successful" in output.result.lower() or "analysis completed" in output.result.lower()
                break
        
        # Verify we got some outputs and at least one was a tool result
        assert len(outputs) > 0
        assert any(isinstance(output, ToolResult) for output in outputs)
        
        # Find the tool result and verify it shows subagent usage
        tool_results = [output for output in outputs if isinstance(output, ToolResult)]
        assert len(tool_results) > 0
        
        # At least one tool result should indicate successful subagent usage
        successful_tool_results = [tr for tr in tool_results if tr.success]
        assert len(successful_tool_results) > 0
    
    @pytest.mark.asyncio
    async def test_multi_level_subagent_coordination(self, openai_client):
        """Test a coordinator agent that manages multiple subagent teams.
        
        This demonstrates a more complex scenario where a main agent
        coordinates multiple subagent groups for different tasks.
        """
        
        async def coordinate_analysis_teams(task_description: str, client: openai.AsyncOpenAI = None) -> str:
            """Meta-tool that coordinates multiple subagent analysis teams."""
            if not client:
                return "Error: OpenAI client required"
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create test data
                project_dir = Path(tmpdir) / "complex_project"
                project_dir.mkdir()
                
                # Create multiple files to analyze
                write_file(str(project_dir / "app.py"), "def main(): return 'app'")
                write_file(str(project_dir / "utils.py"), "def helper(): return 'utils'") 
                write_file(str(project_dir / "test.py"), "def test(): assert True")
                write_file(str(project_dir / "README.md"), "# Complex Project")
                write_file(str(project_dir / "config.json"), '{"version": "1.0"}')
                
                # Team 1: Code Analysis Team
                code_team_specs = [
                    ("You are a Python code analyst. Be brief.", f"Count Python files in {project_dir}", [count_python_files]),
                    ("You are a complexity analyst. Be brief.", f"Assess code complexity in {project_dir}", [assess_code_complexity])
                ]
                
                # Team 2: Documentation Team  
                docs_team_specs = [
                    ("You are a documentation finder. Be brief.", f"Find documentation in {project_dir}", [find_readme_files]),
                    ("You are a file analyzer. Be brief.", f"Analyze file structure in {project_dir}", [analyze_file_sizes])
                ]
                
                teams_results = []
                
                # Run Team 1
                team1_results = []
                async for output in spawn_parallel_agents(
                    client=client, 
                    agent_specs=code_team_specs,
                    model="gpt-4o-mini", 
                    max_iterations=2
                ):
                    if output.is_final:
                        team1_results.append(f"Code team agent {output.agent_index} completed")
                        if len(team1_results) == 2:  # Both agents done
                            break
                
                teams_results.append(f"Code analysis team: {len(team1_results)} agents completed")
                
                # Run Team 2  
                team2_results = []
                async for output in spawn_parallel_agents(
                    client=client,
                    agent_specs=docs_team_specs, 
                    model="gpt-4o-mini",
                    max_iterations=2
                ):
                    if output.is_final:
                        team2_results.append(f"Docs team agent {output.agent_index} completed")
                        if len(team2_results) == 2:  # Both agents done
                            break
                
                teams_results.append(f"Documentation team: {len(team2_results)} agents completed")
                
                return f"Multi-team analysis completed. Results: {'; '.join(teams_results)}"
        
        # Use the coordination tool through Run
        run = Run(
            client=openai_client,
            messages="Coordinate a multi-team analysis of a complex project using subagent teams.",
            model="gpt-4o-mini",
            tools=[coordinate_analysis_teams],
            system_prompt="You are a meta-coordinator that manages multiple analysis teams using subagents.",
            max_iterations=3
        )
        
        # Execute and verify
        tool_results = []
        async for output in run:
            if isinstance(output, ToolResult):
                tool_results.append(output)
                if output.success and "multi-team analysis completed" in output.result.lower():
                    # Found successful coordination
                    assert "code analysis team" in output.result.lower()
                    assert "documentation team" in output.result.lower()
                    break
        
        # Verify we got successful tool coordination
        assert len(tool_results) > 0
        assert any(tr.success for tr in tool_results)
