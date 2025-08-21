# toololo.lib - Standard Library

The `toololo.lib` package provides a set of standard library functions for common operations like file manipulation and shell command execution. Unlike the main toololo framework, these functions do **not** perform git commits automatically.

## Modules

### `toololo.lib.files`

File operations without git integration:

- `read_file(path, include_line_numbers=False, start_line=None, end_line=None)` - Read file contents with optional line number and range support
- `write_file(path, contents)` - Write contents to a new file (fails if exists)
- `str_replace(path, original_content, new_content, replace_all=False)` - Replace string content in files
- `delete_file(path)` - Delete a single file
- `delete_files(paths)` - Delete multiple files
- `make_directory(path)` - Create directory with parents
- `rename_file(old_path, new_path)` - Rename/move files
- `list_directory(path=".", exclude_directories_recursive=None, recursive=False)` - List directory contents

### `toololo.lib.shell`

Shell command execution:

- `shell_command(command, working_directory=".", timeout=600, enable_environment=False, streaming_callback=None)` - Execute shell commands with comprehensive options

Returns a `ShellCommandResult` object with:
- `success` - Boolean indicating if command succeeded
- `returncode` - Exit code
- `stdout` - Standard output
- `stderr` - Standard error
- `execution_time` - Time taken to execute

## Usage Examples

### File Operations

```python
from toololo.lib import files

# Write and read files
files.write_file("config.txt", "server_port=8080\ndebug=false")
content = files.read_file("config.txt")

# Read with line numbers
numbered = files.read_file("config.txt", include_line_numbers=True)

# Read specific line range
section = files.read_file("config.txt", start_line=2, end_line=5)

# String replacement
files.str_replace("config.txt", "debug=false", "debug=true")

# Multiple replacements
files.str_replace("config.txt", "old_text", "new_text", replace_all=True)

# Directory operations
files.make_directory("project/src/utils")
files.list_directory("project", recursive=True)

# File management
files.rename_file("old_name.txt", "new_name.txt")
files.delete_file("unwanted.txt")
files.delete_files(["file1.txt", "file2.txt", "file3.txt"])
```

### Shell Commands

```python
from toololo.lib import shell

# Simple command
result = shell.shell_command("echo 'Hello World'")
if result.success:
    print(result.stdout)

# Command with working directory
result = shell.shell_command("ls -la", working_directory="/tmp")

# Command with timeout
result = shell.shell_command("sleep 10", timeout=5)  # Will timeout

# Enable shell environment (.bashrc, etc.)
result = shell.shell_command("echo $HOME", enable_environment=True)

# Streaming output
def stream_handler(line):
    print(f"Output: {line}")

result = shell.shell_command(
    "for i in {1..5}; do echo $i; sleep 1; done",
    streaming_callback=stream_handler
)
```

### Integration Examples

```python
from toololo.lib import files, shell

# Process files with shell commands
files.write_file("numbers.txt", "5\n2\n8\n1\n9")
result = shell.shell_command("sort -n numbers.txt")
files.write_file("sorted.txt", result.stdout)

# Create and execute scripts
script_content = """#!/bin/bash
echo "Processing files..."
find . -name "*.py" | wc -l
"""
files.write_file("process.sh", script_content)
result = shell.shell_command("chmod +x process.sh && ./process.sh")
```

## Error Handling

All functions raise appropriate Python exceptions:

- `FileNotFoundError` - File or directory doesn't exist
- `FileExistsError` - File already exists when creating
- `PermissionError` - Insufficient permissions
- `IsADirectoryError` - Expected file but got directory
- `NotADirectoryError` - Expected directory but got file
- `ValueError` - Invalid arguments or content not found

```python
from toololo.lib import files, shell

try:
    files.read_file("nonexistent.txt")
except FileNotFoundError as e:
    print(f"File error: {e}")

try:
    files.str_replace("file.txt", "missing", "replacement")
except ValueError as e:
    print(f"Content error: {e}")

# Shell commands return result objects, check success
result = shell.shell_command("nonexistent_command")
if not result.success:
    print(f"Command failed: {result.stderr}")
```

## Key Differences from Main toololo

1. **No Git Integration**: Functions don't automatically commit changes
2. **Simpler Interface**: Direct function calls instead of tool framework
3. **Python Exceptions**: Standard Python error handling instead of tool error messages
4. **No Metadata**: No git commit messages or buffer tracking
5. **Synchronous**: All operations are blocking (except streaming shell commands)

This makes `toololo.lib` ideal for:
- Scripts and automation
- Data processing pipelines
- Testing and experimentation
- Integration with other Python projects
- Situations where git tracking is not desired

## Testing

Run the integration tests:

```bash
python -m pytest test-integration/test_stdlib.py -v
```

See the example script:

```bash
python examples/stdlib_example.py
```
