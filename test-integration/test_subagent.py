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
    """Simple AI tool that generates summaries (mock for testing)."""
    if not client:
        return f"Mock analysis of {focus}: {content[:50]}..." 
    
    # In real use, this would call the OpenAI API
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Analyze this {focus}: {content}"}],
            max_tokens=200
        )
        return response.choices[0].message.content
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


class TestProjectHealthAssessment:
    """Integration test for a creative project health assessment scenario.
    
    This scenario demonstrates multiple specialized agents working together to assess
    the overall health of an open-source project from different perspectives:
    - Codebase Health Agent: Analyzes code structure and complexity
    - Community Health Agent: Checks documentation and project maintenance
    - Infrastructure Agent: Reviews project setup and configuration
    - Security Auditor Agent: Performs basic security checks
    """
    
    @pytest.mark.asyncio
    async def test_project_health_assessment_pipeline(self, openai_client):
        """Test a creative scenario: Multi-agent project health assessment.
        
        This simulates a realistic workflow where different specialized agents
        analyze various aspects of a project simultaneously to provide a 
        comprehensive health assessment.
        """
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a realistic open-source project structure
            project_dir = Path(tmpdir) / "awesome_project"
            await self._create_realistic_project(project_dir)
            
            # Define specialized agents for different aspects of project health
            health_assessment_agents = [
                # Agent 1: Codebase Health Specialist
                (
                    "You are a Codebase Health Specialist. Your expertise is in analyzing code structure, "
                    "complexity, and maintainability. Focus on code quality metrics and organization.",
                    f"Perform a comprehensive codebase health analysis of {project_dir}. "
                    f"Analyze code complexity, file organization, and overall code quality.",
                    [count_python_files, assess_code_complexity, analyze_file_sizes]
                ),
                
                # Agent 2: Community Health Expert
                (
                    "You are a Community Health Expert. You specialize in evaluating project documentation, "
                    "community engagement indicators, and project maintenance signals.",
                    f"Assess the community health and documentation quality of {project_dir}. "
                    f"Look for README files, documentation, and signs of active maintenance.",
                    [find_readme_files, list_directory, ai_summary_tool]
                ),
                
                # Agent 3: Infrastructure Analyst  
                (
                    "You are an Infrastructure Analyst. You focus on project setup, dependencies, "
                    "configuration management, and deployment readiness.",
                    f"Analyze the infrastructure and configuration setup of {project_dir}. "
                    f"Review dependencies, configuration files, and project structure.",
                    [list_directory, read_file, analyze_file_sizes]
                ),
                
                # Agent 4: Security Auditor
                (
                    "You are a Security Auditor. You specialize in identifying potential security issues, "
                    "dependency vulnerabilities, and security best practices.",
                    f"Perform a security assessment of {project_dir}. "
                    f"Check for potential security issues and review project configuration for security.",
                    [check_git_status, read_file, ai_summary_tool]
                )
            ]
            
            # Track results from each specialized agent
            health_report = {
                "codebase": {"agent_name": "Codebase Health", "outputs": [], "status": "running"},
                "community": {"agent_name": "Community Health", "outputs": [], "status": "running"},
                "infrastructure": {"agent_name": "Infrastructure", "outputs": [], "status": "running"},
                "security": {"agent_name": "Security Audit", "outputs": [], "status": "running"}
            }
            
            assessment_areas = ["codebase", "community", "infrastructure", "security"]
            
            print(f"\n🏥 Starting Project Health Assessment for {project_dir.name}")
            print("🤖 Deploying 4 specialized analysis agents...")
            
            # Run all health assessment agents in parallel
            async for result in spawn_parallel_agents(
                client=mock_openai_client,
                agent_specs=health_assessment_agents,
                model="gpt-4",
                max_iterations=8
            ):
                assert isinstance(result, SubagentOutput)
                assert 0 <= result.agent_index <= 3
                
                area = assessment_areas[result.agent_index]
                
                if result.is_final:
                    if result.error:
                        health_report[area]["status"] = f"failed: {result.error}"
                        print(f"❌ {health_report[area]['agent_name']} failed: {result.error}")
                    else:
                        health_report[area]["status"] = "completed"
                        print(f"✅ {health_report[area]['agent_name']} completed")
                else:
                    health_report[area]["outputs"].append(result.output)
                    output_type = type(result.output).__name__
                    print(f"📊 {health_report[area]['agent_name']}: {output_type}")
                
                # Check if all agents completed
                if all(report["status"] != "running" for report in health_report.values()):
                    break
            
            # Verify comprehensive assessment was completed
            print("\n📋 Health Assessment Summary:")
            successful_agents = 0
            total_outputs = 0
            
            for area, report in health_report.items():
                status = "✅ Success" if report["status"] == "completed" else f"❌ {report['status']}"
                output_count = len(report["outputs"])
                total_outputs += output_count
                
                print(f"  {report['agent_name']}: {status} ({output_count} outputs)")
                
                if report["status"] == "completed":
                    successful_agents += 1
                    # Verify each agent produced meaningful outputs
                    assert output_count > 0, f"{area} agent produced no outputs"
            
            # Assertions for test validation
            assert successful_agents >= 3, f"Expected at least 3 successful agents, got {successful_agents}"
            assert total_outputs >= 8, f"Expected at least 8 total outputs, got {total_outputs}"
            
            # Generate final comprehensive health report
            print(f"\n📄 Generating comprehensive project health report...")
            health_score = (successful_agents / 4) * 100
            
            final_report = {
                "project_name": project_dir.name,
                "assessment_date": "test_run",
                "overall_health_score": health_score,
                "agents_deployed": len(health_assessment_agents),
                "successful_assessments": successful_agents,
                "total_analysis_outputs": total_outputs,
                "detailed_results": health_report
            }
            
            print(f"🎯 Overall Project Health Score: {health_score}%")
            print(f"📊 Total Analysis Outputs: {total_outputs}")
            print(f"🤖 Successful Agent Assessments: {successful_agents}/{len(health_assessment_agents)}")
            
            # Verify the assessment was comprehensive
            assert health_score >= 75.0, "Project health assessment should achieve at least 75% success rate"
            
            return final_report
    
    async def _create_realistic_project(self, project_dir: Path):
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
