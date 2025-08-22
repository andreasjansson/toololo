"""Example tools for testing subagent functionality."""

import json
import openai
from typing import Optional
from .files import read_file, list_directory, write_file
from .shell import shell_command


async def analyze_text_with_ai(text: str, analysis_type: str = "summary", client: Optional[openai.AsyncOpenAI] = None) -> str:
    """Analyze text using AI with different analysis types.
    
    Args:
        text: Text to analyze
        analysis_type: Type of analysis (summary, sentiment, keywords, etc.)
        client: OpenAI client (will be bound automatically)
        
    Returns:
        Analysis result as string
    """
    if not client:
        raise ValueError("OpenAI client is required")
    
    prompts = {
        "summary": f"Please provide a concise summary of the following text:\n\n{text}",
        "sentiment": f"Analyze the sentiment of the following text (positive/negative/neutral):\n\n{text}",
        "keywords": f"Extract the key topics and keywords from the following text:\n\n{text}",
        "critique": f"Provide a critical analysis of the following text:\n\n{text}",
    }
    
    prompt = prompts.get(analysis_type, prompts["summary"])
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during AI analysis: {str(e)}"


def analyze_code_file(file_path: str) -> str:
    """Analyze a code file for structure and issues.
    
    Args:
        file_path: Path to the code file
        
    Returns:
        Analysis of the code file
    """
    try:
        content = read_file(file_path)
        lines = content.split('\n')
        
        analysis = {
            "file_path": file_path,
            "total_lines": len(lines),
            "non_empty_lines": len([l for l in lines if l.strip()]),
            "comment_lines": len([l for l in lines if l.strip().startswith('#')]),
            "import_lines": len([l for l in lines if l.strip().startswith(('import ', 'from '))]),
        }
        
        # Check for common patterns
        if file_path.endswith('.py'):
            analysis["functions"] = len([l for l in lines if l.strip().startswith('def ')])
            analysis["classes"] = len([l for l in lines if l.strip().startswith('class ')])
            analysis["has_main"] = any("if __name__ == '__main__':" in l for l in lines)
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        return f"Error analyzing file {file_path}: {str(e)}"


def find_files_with_pattern(directory: str, pattern: str) -> str:
    """Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: Pattern to search for (shell glob pattern)
        
    Returns:
        List of matching files as JSON string
    """
    try:
        result = shell_command(f"find {directory} -name '{pattern}' -type f")
        if result.success:
            files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            return json.dumps({"found_files": files, "count": len(files)}, indent=2)
        else:
            return json.dumps({"error": result.stderr, "found_files": []}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "found_files": []}, indent=2)


def count_lines_in_files(directory: str, file_extension: str = "*.py") -> str:
    """Count total lines in files with specific extension.
    
    Args:
        directory: Directory to search
        file_extension: File extension pattern (e.g., "*.py", "*.js")
        
    Returns:
        Line count statistics as JSON string
    """
    try:
        # Find files and count lines
        result = shell_command(
            f"find {directory} -name '{file_extension}' -type f -exec wc -l {{}} + | tail -n 1",
            working_directory="."
        )
        
        if result.success:
            total_lines = result.stdout.strip().split()[0] if result.stdout.strip() else "0"
            
            # Also get file count
            file_result = shell_command(f"find {directory} -name '{file_extension}' -type f | wc -l")
            file_count = file_result.stdout.strip() if file_result.success else "0"
            
            return json.dumps({
                "directory": directory,
                "file_extension": file_extension,
                "total_lines": int(total_lines),
                "file_count": int(file_count),
                "avg_lines_per_file": round(int(total_lines) / max(1, int(file_count)), 1)
            }, indent=2)
        else:
            return json.dumps({"error": result.stderr}, indent=2)
            
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def create_project_report(project_dir: str, report_path: str) -> str:
    """Create a comprehensive project report.
    
    Args:
        project_dir: Directory of the project to analyze
        report_path: Path where to save the report
        
    Returns:
        Success message or error
    """
    try:
        # Gather project statistics
        listing = list_directory(project_dir, recursive=True)
        
        # Count different file types
        py_files = shell_command(f"find {project_dir} -name '*.py' -type f | wc -l")
        py_count = int(py_files.stdout.strip()) if py_files.success else 0
        
        js_files = shell_command(f"find {project_dir} -name '*.js' -type f | wc -l")
        js_count = int(js_files.stdout.strip()) if js_files.success else 0
        
        md_files = shell_command(f"find {project_dir} -name '*.md' -type f | wc -l")
        md_count = int(md_files.stdout.strip()) if md_files.success else 0
        
        # Create report content
        report_content = f"""# Project Report: {project_dir}

## File Statistics
- Python files: {py_count}
- JavaScript files: {js_count}
- Markdown files: {md_count}

## Directory Structure
```
{listing}
```

## Generated at
{shell_command('date').stdout.strip()}
"""
        
        # Write report
        write_file(report_path, report_content)
        return f"Project report created successfully at {report_path}"
        
    except Exception as e:
        return f"Error creating project report: {str(e)}"
