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


# Simple tools for the demo
def count_python_files(directory: str) -> str:
    """Count Python files in a directory."""
    result = shell_command(f"find {directory} -name '*.py' -type f | wc -l")
    if result.success:
        count = int(result.stdout.strip())
        return f"Found {count} Python files in {directory}"
    return f"Error counting files: {result.stderr}"


def find_readme_files(directory: str) -> str:
    """Find README files."""
    result = shell_command(f"find {directory} -iname 'readme*' -type f")
    if result.success:
        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        if files:
            return f"Found {len(files)} README files: {', '.join([Path(f).name for f in files])}"
        return "No README files found"
    return f"Error searching: {result.stderr}"


def check_project_structure(directory: str) -> str:
    """Check basic project structure."""
    structure_score = 0
    findings = []
    
    # Check for common directories
    common_dirs = ['src', 'lib', 'tests', 'test', 'docs', 'doc']
    for dirname in common_dirs:
        if Path(directory, dirname).exists():
            structure_score += 10
            findings.append(f"+ Found {dirname}/ directory")
    
    # Check for important files
    important_files = ['README.md', 'README.txt', 'requirements.txt', 'setup.py', 'pyproject.toml']
    for filename in important_files:
        if Path(directory, filename).exists():
            structure_score += 10
            findings.append(f"+ Found {filename}")
    
    return f"Structure score: {structure_score}/100. " + "; ".join(findings[:3])


async def main():
    """Main demo function."""
    print("🏥 Creative Project Health Assessment Demo")
    print("=" * 50)
    print("This demonstrates multiple specialized AI agents working together")
    print("to assess different aspects of a project's health.\n")
    
    # Create OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("💡 Set your API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    client = openai.AsyncOpenAI(api_key=api_key)
    
    # Create a realistic sample project for analysis
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir) / "awesome_project"
        project_dir.mkdir()
        
        print(f"📁 Creating sample project at: {project_dir.name}")
        
        # Create realistic project structure
        await create_demo_project(project_dir)
        
        # Define 4 specialized health assessment agents
        health_agents = [
            # Agent 1: Codebase Health Specialist
            (
                "You are a Codebase Health Specialist. Analyze code structure, "
                "organization, and quality metrics. Be concise and specific.",
                f"Analyze the codebase health of {project_dir}. "
                f"Focus on code organization, file counts, and structure quality.",
                [count_python_files, check_project_structure, list_directory]
            ),
            
            # Agent 2: Documentation Health Expert
            (
                "You are a Documentation Health Expert. Evaluate documentation "
                "completeness and quality. Be concise and actionable.",
                f"Assess the documentation health of {project_dir}. "
                f"Look for README files, docs, and documentation completeness.",
                [find_readme_files, list_directory, read_file]
            ),
            
            # Agent 3: Project Structure Analyst
            (
                "You are a Project Structure Analyst. Evaluate project organization "
                "and best practices adherence. Be specific about findings.",
                f"Evaluate the project structure and organization of {project_dir}. "
                f"Check for standard directories and file organization patterns.",
                [check_project_structure, list_directory, count_python_files]
            ),
            
            # Agent 4: Community Health Assessor
            (
                "You are a Community Health Assessor. Look for signs of active "
                "maintenance and community engagement. Be observational.",
                f"Assess community health indicators in {project_dir}. "
                f"Look for signs of maintenance, configuration, and project health.",
                [find_readme_files, list_directory, read_file]
            )
        ]
        
        print(f"\n🚀 Deploying {len(health_agents)} specialized health assessment agents...")
        
        # Track health assessment results
        health_report = {}
        agent_names = ["Codebase Health", "Documentation Health", "Structure Analysis", "Community Health"]
        
        # Run health assessment agents in parallel
        async for result in spawn_parallel_agents(
            client=client,
            agent_specs=health_agents,
            model="gpt-4o-mini",  # Use faster, cheaper model for demo
            max_iterations=3
        ):
            agent_idx = result.agent_index
            agent_name = agent_names[agent_idx]
            
            if agent_idx not in health_report:
                health_report[agent_idx] = {
                    "name": agent_name,
                    "outputs": [],
                    "completed": False,
                    "error": None
                }
            
            if result.is_final:
                health_report[agent_idx]["completed"] = True
                if result.error:
                    health_report[agent_idx]["error"] = result.error
                    print(f"❌ {agent_name}: Failed - {result.error}")
                else:
                    print(f"✅ {agent_name}: Assessment completed")
            else:
                health_report[agent_idx]["outputs"].append(result.output)
                output_type = type(result.output).__name__
                print(f"📊 {agent_name}: {output_type}")
            
            # Check if all agents completed
            if all(health_report.get(i, {}).get("completed", False) for i in range(len(health_agents))):
                break
        
        # Generate Health Assessment Report
        print(f"\n🏥 PROJECT HEALTH ASSESSMENT REPORT")
        print("=" * 50)
        
        successful_assessments = 0
        total_outputs = 0
        
        for i, assessment in health_report.items():
            name = assessment["name"]
            output_count = len(assessment["outputs"])
            total_outputs += output_count
            
            if assessment["completed"] and not assessment["error"]:
                successful_assessments += 1
                status = "✅ HEALTHY"
            else:
                status = f"⚠️ ISSUES: {assessment.get('error', 'Unknown error')}"
            
            print(f"\n{name}:")
            print(f"  Status: {status}")
            print(f"  Analysis Depth: {output_count} tool outputs")
        
        # Overall health score
        health_score = (successful_assessments / len(health_agents)) * 100
        print(f"\n🎯 OVERALL PROJECT HEALTH SCORE: {health_score:.0f}%")
        print(f"📊 Successful Assessments: {successful_assessments}/{len(health_agents)}")
        print(f"🔧 Total Analysis Operations: {total_outputs}")
        
        if health_score >= 75:
            print("🌟 PROJECT STATUS: HEALTHY")
        elif health_score >= 50:
            print("⚠️ PROJECT STATUS: MODERATE HEALTH")
        else:
            print("🚨 PROJECT STATUS: NEEDS ATTENTION")
        
        print(f"\n🎉 Multi-agent health assessment completed!")
        print(f"💡 This demo showed {len(health_agents)} AI agents working in parallel")
        print(f"   to assess different aspects of project health simultaneously.")


async def create_demo_project(project_dir: Path):
    """Create a realistic demo project structure."""
    # Create directories
    (project_dir / "src").mkdir()
    (project_dir / "tests").mkdir()  
    (project_dir / "docs").mkdir()
    
    # Create essential project files
    project_files = {
        "README.md": "# Awesome Project\nA demo project for health assessment.\n\n## Features\n- Clean code\n- Good docs\n- Tests",
        "src/main.py": "#!/usr/bin/env python3\ndef main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()",
        "src/utils.py": "def helper():\n    return 'help'\n\nclass Utils:\n    def process(self):\n        pass",
        "tests/test_main.py": "import unittest\n\nclass TestMain(unittest.TestCase):\n    def test_main(self):\n        pass",
        "requirements.txt": "click>=8.0.0\nrequests>=2.28.0",
        "docs/README.md": "# Documentation\nProject docs go here."
    }
    
    # Write files
    for file_path, content in project_files.items():
        full_path = project_dir / file_path
        write_file(str(full_path), content)
    
    print(f"✅ Created {len(project_files)} files in project structure")
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
