"""Integration tests for toololo.lib.subagent module."""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import openai
from toololo.lib.subagent import spawn_parallel_agents, SubagentOutput, ParallelSubagents
from toololo.lib.files import write_file, read_file, list_directory
from toololo.lib.shell import shell_command
from toololo.lib.example_tools import (
    analyze_text_with_ai, analyze_code_file, find_files_with_pattern,
    count_lines_in_files, create_project_report
)
from toololo.types import TextContent, ToolUseContent, ToolResult


# Mock OpenAI client for testing
class MockOpenAI:
    """Mock OpenAI client for testing."""
    
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.call_count = 0
        self.responses = []
    
    def set_responses(self, responses):
        """Set predefined responses for the mock."""
        self.responses = responses
        self.call_count = 0
    
    async def create(self, **kwargs):
        """Mock create method that returns predefined responses."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        
        # Default response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = f"Mock AI response {self.call_count}"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        self.call_count += 1
        return mock_response


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = MockOpenAI()
    client.chat.completions.create = client.create
    return client


class TestSubagentTools:
    """Test the example tools used by subagents."""
    
    def test_analyze_code_file(self):
        """Test code analysis tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample Python file
            py_file = Path(tmpdir) / "sample.py"
            code_content = '''#!/usr/bin/env python3
"""Sample Python module."""

import os
import sys
from pathlib import Path

class TestClass:
    """A test class."""
    
    def method1(self):
        """Test method."""
        pass
    
    def method2(self):
        # This is a comment
        return "test"

def main():
    """Main function."""
    print("Hello World")

if __name__ == "__main__":
    main()
'''
            write_file(str(py_file), code_content)
            
            # Analyze the file
            result = analyze_code_file(str(py_file))
            analysis = json.loads(result)
            
            assert analysis["file_path"] == str(py_file)
            assert analysis["total_lines"] > 20
            assert analysis["functions"] >= 2  # main and methods
            assert analysis["classes"] >= 1
            assert analysis["has_main"] is True
            assert analysis["import_lines"] >= 3
    
    def test_find_files_with_pattern(self):
        """Test file pattern matching tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            files_to_create = ["test1.py", "test2.py", "readme.md", "config.json"]
            for filename in files_to_create:
                write_file(str(Path(tmpdir) / filename), f"Content of {filename}")
            
            # Find Python files
            result = find_files_with_pattern(tmpdir, "*.py")
            data = json.loads(result)
            
            assert data["count"] == 2
            assert len(data["found_files"]) == 2
            assert all(f.endswith(".py") for f in data["found_files"])
    
    def test_count_lines_in_files(self):
        """Test line counting tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Python files with known line counts
            py_file1 = Path(tmpdir) / "file1.py"
            py_file2 = Path(tmpdir) / "file2.py"
            
            write_file(str(py_file1), "line1\nline2\nline3")  # 3 lines
            write_file(str(py_file2), "line1\nline2")         # 2 lines
            
            result = count_lines_in_files(tmpdir, "*.py")
            data = json.loads(result)
            
            assert data["file_count"] == 2
            assert data["total_lines"] == 5
            assert data["avg_lines_per_file"] == 2.5
    
    def test_create_project_report(self):
        """Test project report creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            
            # Create project structure
            (project_dir / "src").mkdir()
            write_file(str(project_dir / "README.md"), "# Test Project")
            write_file(str(project_dir / "src" / "main.py"), "print('hello')")
            write_file(str(project_dir / "src" / "utils.js"), "console.log('test')")
            
            # Create report
            report_path = Path(tmpdir) / "report.md"
            result = create_project_report(str(project_dir), str(report_path))
            
            assert "successfully" in result
            assert Path(report_path).exists()
            
            # Check report content
            report_content = read_file(str(report_path))
            assert "# Project Report" in report_content
            assert "Python files: 1" in report_content
            assert "JavaScript files: 1" in report_content
            assert "Markdown files: 1" in report_content

    @pytest.mark.asyncio
    async def test_analyze_text_with_ai(self, mock_openai_client):
        """Test AI text analysis tool."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a positive text about success."
        mock_openai_client.set_responses([mock_response])
        
        # Test the tool
        result = await analyze_text_with_ai(
            "I love this amazing project!",
            "sentiment",
            client=mock_openai_client
        )
        
        assert "positive" in result.lower()
        assert mock_openai_client.call_count == 1


class TestSubagentCore:
    """Test core subagent functionality."""
    
    def test_parallel_subagents_init(self, mock_openai_client):
        """Test ParallelSubagents initialization."""
        manager = ParallelSubagents(
            client=mock_openai_client,
            model="gpt-4",
            max_tokens=1000
        )
        
        assert manager.client == mock_openai_client
        assert manager.model == "gpt-4"
        assert manager.max_tokens == 1000
        assert manager._agents == []
    
    def test_bind_client_to_tools(self, mock_openai_client):
        """Test client binding to tools."""
        manager = ParallelSubagents(client=mock_openai_client)
        
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
        assert "MockOpenAI" in result
        
        # Test the simple tool (unchanged)
        result = bound_tools[1]("test")
        assert result == "Simple: test"


class TestIntegrationScenarios:
    """Integration tests with realistic subagent scenarios."""
    
    @pytest.mark.asyncio
    async def test_parallel_code_analysis_scenario(self, mock_openai_client):
        """Test a realistic scenario: parallel code analysis of a project."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            
            # Create a realistic project structure
            (project_dir / "src").mkdir()
            (project_dir / "tests").mkdir()
            (project_dir / "docs").mkdir()
            
            # Create various files
            files_content = {
                "README.md": "# My Awesome Project\nThis is a test project for analysis.",
                "src/main.py": '''#!/usr/bin/env python3
"""Main application module."""
import argparse
import sys

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="My app")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    if args.verbose:
        print("Running in verbose mode")
    
    print("Hello, World!")

if __name__ == "__main__":
    main()
''',
                "src/utils.py": '''"""Utility functions."""

def helper_function(data):
    """Process data."""
    return [x * 2 for x in data]

def another_helper(text):
    """Process text."""
    return text.upper()

class DataProcessor:
    """Data processing class."""
    
    def __init__(self):
        self.data = []
    
    def process(self, items):
        """Process items."""
        return [helper_function(item) for item in items]
''',
                "tests/test_main.py": '''"""Tests for main module."""
import unittest
from src.main import main

class TestMain(unittest.TestCase):
    """Test main functionality."""
    
    def test_main_runs(self):
        """Test that main runs without error."""
        # Would test main function here
        pass

if __name__ == "__main__":
    unittest.main()
''',
                "docs/architecture.md": "# Architecture\n\nThis document describes the system architecture."
            }
            
            for file_path, content in files_content.items():
                full_path = project_dir / file_path
                write_file(str(full_path), content)
            
            # Set up mock AI responses
            ai_responses = [
                # Response for structure analyzer
                self._create_mock_response(
                    f"The project has a clean structure with {len(files_content)} files organized into src/, tests/, and docs/ directories."
                ),
                # Response for code quality analyzer  
                self._create_mock_response(
                    "The Python code follows good practices with proper docstrings, main guards, and clear function definitions."
                ),
                # Response for documentation analyzer
                self._create_mock_response(
                    "The documentation includes README and architecture docs, providing good project overview."
                )
            ]
            mock_openai_client.set_responses(ai_responses)
            
            # Define agent specifications for parallel analysis
            agent_specs = [
                # Agent 1: Analyze project structure
                (
                    "You are a project structure analyzer. Analyze the directory structure and file organization of projects.",
                    f"Analyze the structure of the project at {project_dir}. List the directories and key files.",
                    [list_directory, find_files_with_pattern, analyze_text_with_ai]
                ),
                # Agent 2: Analyze code quality
                (
                    "You are a code quality analyzer. Review code files for best practices and quality metrics.",
                    f"Analyze the Python code quality in {project_dir}. Focus on code structure, functions, and classes.",
                    [analyze_code_file, count_lines_in_files, find_files_with_pattern]
                ),
                # Agent 3: Analyze documentation
                (
                    "You are a documentation analyzer. Review project documentation for completeness and quality.",
                    f"Analyze the documentation in {project_dir}. Review README and other docs.",
                    [read_file, find_files_with_pattern, analyze_text_with_ai]
                )
            ]
            
            # Track outputs from all agents
            outputs_by_agent = {0: [], 1: [], 2: []}
            completed_agents = set()
            
            # Run parallel analysis
            async for result in spawn_parallel_agents(
                client=mock_openai_client,
                agent_specs=agent_specs,
                model="gpt-4",
                max_iterations=3
            ):
                assert isinstance(result, SubagentOutput)
                assert 0 <= result.agent_index <= 2
                assert result.agent_id.startswith(f"agent_{result.agent_index}_")
                
                if result.is_final:
                    completed_agents.add(result.agent_index)
                    if result.error:
                        print(f"Agent {result.agent_index} failed: {result.error}")
                else:
                    outputs_by_agent[result.agent_index].append(result.output)
                
                # Stop when all agents complete
                if len(completed_agents) == 3:
                    break
            
            # Verify all agents completed
            assert len(completed_agents) == 3
            
            # Verify we got outputs from each agent
            for agent_idx in range(3):
                assert len(outputs_by_agent[agent_idx]) > 0, f"Agent {agent_idx} produced no outputs"
    
    @pytest.mark.asyncio
    async def test_collaborative_report_generation(self, mock_openai_client):
        """Test agents collaborating to generate a comprehensive report."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "sample_project"
            project_dir.mkdir()
            
            # Create sample project files
            project_files = {
                "app.py": "print('Main app')\ndef process_data(): pass\nclass DataHandler: pass",
                "utils.py": "def helper(): return 'help'\ndef formatter(x): return str(x)",
                "config.json": '{"version": "1.0", "debug": true}',
                "README.md": "# Sample Project\nA simple demonstration project.",
                "requirements.txt": "requests==2.25.1\nnumpy==1.21.0"
            }
            
            for filename, content in project_files.items():
                write_file(str(project_dir / filename), content)
            
            # Set up mock responses for different analysis tasks
            mock_responses = [
                self._create_mock_response("Project has 5 files with good organization."),
                self._create_mock_response("Code quality is good with 2 Python files containing 3 functions and 1 class."),
                self._create_mock_response("Dependencies include requests and numpy. No security issues found.")
            ]
            mock_openai_client.set_responses(mock_responses)
            
            # Define collaborative agents
            agent_specs = [
                # File structure analyzer
                (
                    "You are a file structure specialist. Analyze project files and organization.",
                    f"Examine the file structure of {project_dir} and create a summary.",
                    [list_directory, find_files_with_pattern, count_lines_in_files]
                ),
                # Code analyzer
                (
                    "You are a code analysis expert. Analyze code files for metrics and quality.",
                    f"Analyze all Python files in {project_dir} for code metrics.",
                    [analyze_code_file, find_files_with_pattern]
                ),
                # Dependency analyzer
                (
                    "You are a dependency and security analyst. Review project dependencies.",
                    f"Analyze dependencies and configuration files in {project_dir}.",
                    [read_file, find_files_with_pattern, analyze_text_with_ai]
                )
            ]
            
            # Collect all outputs for final report
            agent_results = {}
            tool_calls_seen = 0
            
            async for result in spawn_parallel_agents(
                client=mock_openai_client,
                agent_specs=agent_specs,
                max_iterations=5
            ):
                agent_idx = result.agent_index
                
                if agent_idx not in agent_results:
                    agent_results[agent_idx] = {"outputs": [], "completed": False}
                
                if result.is_final:
                    agent_results[agent_idx]["completed"] = True
                    if result.error:
                        agent_results[agent_idx]["error"] = result.error
                else:
                    agent_results[agent_idx]["outputs"].append(result.output)
                    # Count tool calls to verify agents are working
                    if isinstance(result.output, ToolResult):
                        tool_calls_seen += 1
                
                # Break when all agents are done
                if all(agent_results.get(i, {}).get("completed", False) for i in range(3)):
                    break
            
            # Verify all agents completed successfully
            assert len(agent_results) == 3
            for i in range(3):
                assert agent_results[i]["completed"]
                assert len(agent_results[i]["outputs"]) > 0
                assert "error" not in agent_results[i]
            
            # Verify tool calls were made
            assert tool_calls_seen > 0, "Expected to see tool calls from agents"
            
            # Generate collaborative report
            report_path = Path(tmpdir) / "collaborative_report.md"
            report_result = create_project_report(str(project_dir), str(report_path))
            
            assert "successfully" in report_result
            assert report_path.exists()
            
            # Verify report content
            report_content = read_file(str(report_path))
            assert "Project Report" in report_content
            assert str(project_dir) in report_content
    
    def _create_mock_response(self, content: str):
        """Helper to create mock OpenAI responses."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        return mock_response


class TestErrorHandling:
    """Test error handling in subagent scenarios."""
    
    @pytest.mark.asyncio
    async def test_agent_with_failing_tools(self, mock_openai_client):
        """Test handling of agents with tools that fail."""
        
        def failing_tool(input_text: str) -> str:
            """A tool that always fails."""
            raise ValueError("This tool always fails!")
        
        def working_tool(input_text: str) -> str:
            """A tool that works."""
            return f"Processed: {input_text}"
        
        # Set up a mock response that would try to use tools
        mock_openai_client.set_responses([
            self._create_mock_response("I'll analyze the input using my tools.")
        ])
        
        agent_specs = [
            (
                "You are a test agent with mixed tools.",
                "Process the text 'test input'",
                [working_tool, failing_tool]
            )
        ]
        
        results = []
        async for result in spawn_parallel_agents(
            client=mock_openai_client,
            agent_specs=agent_specs,
            max_iterations=3
        ):
            results.append(result)
            if result.is_final:
                break
        
        # Should have completed (though may have had tool failures)
        assert any(r.is_final for r in results)
        final_result = [r for r in results if r.is_final][0]
        assert final_result.agent_index == 0
    
    @pytest.mark.asyncio  
    async def test_agent_timeout_scenario(self, mock_openai_client):
        """Test agent behavior with very low iteration limits."""
        
        # Mock response
        mock_openai_client.set_responses([
            self._create_mock_response("Starting analysis...")
        ])
        
        agent_specs = [
            (
                "You are a simple agent.",
                "Say hello",
                [lambda: "Hello World"]
            )
        ]
        
        results = []
        async for result in spawn_parallel_agents(
            client=mock_openai_client,
            agent_specs=agent_specs,
            max_iterations=1  # Very low limit
        ):
            results.append(result)
            if result.is_final:
                break
        
        # Should complete quickly with low iteration limit
        assert any(r.is_final for r in results)
    
    def _create_mock_response(self, content: str):
        """Helper to create mock OpenAI responses."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        return mock_response
