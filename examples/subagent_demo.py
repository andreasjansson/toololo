#!/usr/bin/env python3
"""
Creative Project Health Assessment Demo

This demonstrates a realistic and creative use case for parallel subagents:
Multiple specialized AI agents working together to assess the overall health
of an open-source project from different perspectives.

Agents:
- 🔍 Codebase Health Analyst: Code quality and structure
- 👥 Community Health Expert: Documentation and maintenance  
- 🏗️ Infrastructure Specialist: Setup and configuration
- 🔒 Security Auditor: Security and vulnerability assessment
"""

import asyncio
import tempfile
import os
from pathlib import Path

import openai
from toololo.lib.subagent import spawn_parallel_agents
from toololo.lib.files import write_file, list_directory, read_file
from toololo.lib.shell import shell_command


async def main():
    """Main demo function."""
    print("🤖 Parallel Subagent Demo")
    print("=" * 50)
    
    # Create a mock client (replace with real OpenAI client in production)
    # client = openai.AsyncOpenAI(api_key="your-api-key")
    client = DemoOpenAIClient()
    client.chat.completions.create = client.create
    
    # Create a sample project for analysis
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir) / "demo_project"
        project_dir.mkdir()
        
        print(f"📁 Created demo project at: {project_dir}")
        
        # Create realistic project structure
        await create_demo_project(project_dir)
        
        # Define specialized agents
        agent_specs = [
            # Agent 1: Structure Analyzer
            (
                "You are a project structure analyst. Your role is to examine project "
                "organization, directory structure, and file distribution. Provide insights "
                "about how well the project is organized.",
                f"Analyze the structure and organization of the project at {project_dir}. "
                f"Examine the directory layout and file distribution.",
                [list_directory, find_files_with_pattern, count_lines_in_files]
            ),
            
            # Agent 2: Code Quality Analyst  
            (
                "You are a code quality specialist. Your role is to analyze code files "
                "for metrics, complexity, and best practices. Focus on Python code analysis.",
                f"Perform a code quality analysis of the Python files in {project_dir}. "
                f"Examine code structure, functions, classes, and overall quality.",
                [analyze_code_file, find_files_with_pattern, count_lines_in_files]
            ),
            
            # Agent 3: Documentation Analyst
            (
                "You are a documentation analyst. Your role is to evaluate project "
                "documentation for completeness, clarity, and usefulness.",
                f"Analyze the documentation in {project_dir}. Review README files, "
                f"comments, and any other documentation for quality and completeness.",
                [find_files_with_pattern, analyze_text_with_ai]
            )
        ]
        
        print(f"\n🚀 Launching {len(agent_specs)} parallel agents...")
        
        # Track results from each agent
        agent_results = {}
        agent_names = ["Structure Analyst", "Code Quality Analyst", "Documentation Analyst"]
        
        # Run agents in parallel and collect results
        async for result in spawn_parallel_agents(
            client=client,
            agent_specs=agent_specs,
            model="gpt-4",
            max_iterations=5
        ):
            agent_idx = result.agent_index
            agent_name = agent_names[agent_idx]
            
            if agent_idx not in agent_results:
                agent_results[agent_idx] = {
                    "name": agent_name,
                    "outputs": [],
                    "completed": False,
                    "error": None
                }
            
            if result.is_final:
                agent_results[agent_idx]["completed"] = True
                if result.error:
                    agent_results[agent_idx]["error"] = result.error
                    print(f"❌ {agent_name} failed: {result.error}")
                else:
                    print(f"✅ {agent_name} completed successfully")
            else:
                agent_results[agent_idx]["outputs"].append(result.output)
                output_type = type(result.output).__name__
                print(f"📊 {agent_name}: {output_type}")
            
            # Check if all agents completed
            if all(agent_results.get(i, {}).get("completed", False) for i in range(len(agent_specs))):
                break
        
        print("\n📋 Final Results Summary:")
        print("-" * 40)
        
        for i, agent_result in agent_results.items():
            name = agent_result["name"]
            output_count = len(agent_result["outputs"])
            status = "✅ Success" if agent_result["completed"] and not agent_result["error"] else "❌ Failed"
            
            print(f"{name}:")
            print(f"  Status: {status}")
            print(f"  Outputs: {output_count}")
            
            if agent_result["error"]:
                print(f"  Error: {agent_result['error']}")
            
            # Show some sample outputs
            if output_count > 0:
                print(f"  Sample outputs:")
                for j, output in enumerate(agent_result["outputs"][:2]):  # Show first 2
                    output_type = type(output).__name__
                    print(f"    {j+1}. {output_type}")
        
        # Generate final collaborative report
        print(f"\n📄 Generating collaborative project report...")
        report_path = project_dir.parent / "final_report.md"
        report_result = create_project_report(str(project_dir), str(report_path))
        print(f"Report: {report_result}")
        
        print(f"\n🎉 Demo completed! All {len(agent_specs)} agents finished their analysis.")
        print(f"📊 Total outputs collected: {sum(len(r['outputs']) for r in agent_results.values())}")


async def create_demo_project(project_dir: Path):
    """Create a realistic demo project structure."""
    print("🏗️ Creating demo project structure...")
    
    # Create directories
    (project_dir / "src").mkdir()
    (project_dir / "tests").mkdir()  
    (project_dir / "docs").mkdir()
    (project_dir / "config").mkdir()
    
    # Create various file types
    files_content = {
        "README.md": """# Demo Project

This is a demonstration project for testing parallel subagent analysis.

## Features
- Multi-module Python application
- Comprehensive test suite
- Documentation
- Configuration management

## Usage
Run the main application with:
```bash
python src/main.py
```
""",
        
        "src/main.py": """#!/usr/bin/env python3
\"\"\"Main application entry point.\"\"\"

import argparse
import logging
from pathlib import Path

from .data_processor import DataProcessor
from .utils import setup_logging, load_config


def main():
    \"\"\"Main application function.\"\"\"
    parser = argparse.ArgumentParser(description="Demo Application")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("input_file", help="Input file to process")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Process data
    processor = DataProcessor(config)
    result = processor.process_file(args.input_file)
    
    print(f"Processing completed: {result}")


if __name__ == "__main__":
    main()
""",
        
        "src/data_processor.py": """\"\"\"Data processing module.\"\"\"

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class DataProcessor:
    \"\"\"Main data processing class.\"\"\"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processed_count = 0
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        \"\"\"Process a single file.\"\"\"
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        logger.info(f"Processing file: {file_path}")
        
        # Simulate processing based on file type
        if path.suffix == '.json':
            return self._process_json(path)
        elif path.suffix == '.txt':
            return self._process_text(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def _process_json(self, path: Path) -> Dict[str, Any]:
        \"\"\"Process JSON files.\"\"\"
        with open(path, 'r') as f:
            data = json.load(f)
        
        result = {
            'type': 'json',
            'records': len(data) if isinstance(data, list) else 1,
            'size_bytes': path.stat().st_size
        }
        
        self.processed_count += 1
        return result
    
    def _process_text(self, path: Path) -> Dict[str, Any]:
        \"\"\"Process text files.\"\"\"
        with open(path, 'r') as f:
            content = f.read()
        
        result = {
            'type': 'text',
            'lines': len(content.splitlines()),
            'words': len(content.split()),
            'characters': len(content)
        }
        
        self.processed_count += 1
        return result
""",
        
        "src/utils.py": """\"\"\"Utility functions.\"\"\"

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging(level: int = logging.INFO) -> None:
    \"\"\"Setup application logging.\"\"\"
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    \"\"\"Load application configuration.\"\"\"
    default_config = {
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'supported_formats': ['json', 'txt'],
        'output_format': 'json'
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
        
        # Merge with defaults
        default_config.update(user_config)
        return default_config
    
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file: {e}")
        return default_config


def format_bytes(bytes_count: int) -> str:
    \"\"\"Format byte count in human readable form.\"\"\"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} TB"
""",
        
        "tests/test_data_processor.py": """\"\"\"Tests for data processor module.\"\"\"

import json
import tempfile
import unittest
from pathlib import Path

from src.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    \"\"\"Test cases for DataProcessor.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test fixtures.\"\"\"
        self.config = {
            'max_file_size': 1024,
            'supported_formats': ['json', 'txt']
        }
        self.processor = DataProcessor(self.config)
    
    def test_process_json_file(self):
        \"\"\"Test processing JSON files.\"\"\"
        test_data = [{'id': 1, 'name': 'test'}, {'id': 2, 'name': 'test2'}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = self.processor.process_file(temp_path)
            self.assertEqual(result['type'], 'json')
            self.assertEqual(result['records'], 2)
            self.assertGreater(result['size_bytes'], 0)
        finally:
            Path(temp_path).unlink()
    
    def test_process_text_file(self):
        \"\"\"Test processing text files.\"\"\"
        test_content = "Hello world\\nThis is a test file\\nWith multiple lines"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = self.processor.process_file(temp_path)
            self.assertEqual(result['type'], 'text')
            self.assertEqual(result['lines'], 3)
            self.assertGreater(result['words'], 0)
        finally:
            Path(temp_path).unlink()
    
    def test_file_not_found(self):
        \"\"\"Test handling of missing files.\"\"\"
        with self.assertRaises(FileNotFoundError):
            self.processor.process_file('/nonexistent/file.json')


if __name__ == '__main__':
    unittest.main()
""",
        
        "docs/api.md": """# API Documentation

## DataProcessor Class

The `DataProcessor` class is the main component for processing various file formats.

### Constructor

```python
DataProcessor(config: Dict[str, Any])
```

Initialize the processor with configuration options.

### Methods

#### process_file(file_path: str) -> Dict[str, Any]

Process a single file and return processing results.

**Parameters:**
- `file_path`: Path to the file to process

**Returns:**
- Dictionary containing processing results and metadata

**Raises:**
- `FileNotFoundError`: If the specified file doesn't exist
- `ValueError`: If the file type is not supported
""",
        
        "config/default.json": """{
    "max_file_size": 10485760,
    "supported_formats": ["json", "txt", "csv"],
    "output_format": "json",
    "logging_level": "INFO",
    "enable_metrics": true
}"""
    }
    
    # Write all files
    for file_path, content in files_content.items():
        full_path = project_dir / file_path
        write_file(str(full_path), content)
    
    print(f"✅ Created {len(files_content)} files across multiple directories")


if __name__ == "__main__":
    asyncio.run(main())
